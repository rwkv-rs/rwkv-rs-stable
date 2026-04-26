use burn::{
    backend::autodiff::{
        Autodiff,
        NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::ops::FloatTensor,
};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    CubeTuneId,
    FloatElement,
    IntElement,
    cubecl::{
        CubeCount,
        CubeDim,
        calculate_cube_count_elemwise,
        prelude::*,
        tensor_vector_size_parallel,
        tune::{
            AsFunctionTunable,
            AutotuneKey,
            AutotuneOutput,
            LocalTuner,
            Tunable,
            TunableSet,
            TuneGroup,
            anchor,
            local_tuner,
        },
    },
    element::BoolElement,
    ops::numeric::{empty_device, zeros_client},
    tensor::CubeTensor,
};
use serde::{Deserialize, Serialize};

use crate::kernels::addcmul::{
    AddcmulBackend,
    io::{
        Addcmul5ForwardPrimitiveInputs,
        Addcmul5ForwardPrimitiveOutput,
        AddcmulBackwardPrimitiveOutputs,
        AddcmulForwardPrimitiveInputs,
    },
    kernel::{
        addcmul_backward_diff_kernel,
        addcmul_scale_reduce_finalize_kernel,
        addcmul_scale_reduce_partial_kernel,
    },
};

impl<R, F, I, BT, C> AddcmulBackend for Autodiff<CubeBackend<R, F, I, BT>, C>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
    C: CheckpointStrategy,
{
    fn fused_addcmul(inputs: AddcmulForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct FusedAddcmulBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 3> for FusedAddcmulBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = (NodeId, FloatTensor<CubeBackend<R, F, I, BT>>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_base, node_diff, node_scale] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let (diff_state, scale) = ops.state;
                let diff: FloatTensor<CubeBackend<R, F, I, BT>> =
                    checkpointer.retrieve_node_output(diff_state);

                assert!(diff.is_contiguous(), "diff must be contiguous");
                assert!(scale.is_contiguous(), "scale must be contiguous");
                assert!(
                    output_grad.is_contiguous(),
                    "output_grad must be contiguous"
                );

                let client = diff.client.clone();
                let key =
                    |diff: &CubeTensor<R>, scale: &CubeTensor<R>, output_grad: &CubeTensor<R>| {
                        let shape = diff.meta.shape();
                        let max_line_size = [&diff, &scale, &output_grad]
                            .into_iter()
                            .map(|tensor| {
                                tensor_vector_size_parallel(
                                    tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                                    tensor.meta.shape(),
                                    tensor.meta.strides(),
                                    shape.num_dims() - 1,
                                )
                            })
                            .min()
                            .unwrap_or(1)
                            .max(1);

                        AddcmulBackwardAutotuneKey {
                            dtype: diff.dtype,
                            num_elements: anchor(shape.num_elements(), None, Some(1), None),
                            embedded_dim: shape[2],
                            bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                            max_line_size,
                        }
                    };

                let input_gen = |_key: &AddcmulBackwardAutotuneKey,
                                 diff: &CubeTensor<R>,
                                 scale: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                    (diff.copy(), scale.copy(), output_grad.copy())
                };

                static TUNER: LocalTuner<AddcmulBackwardAutotuneKey, CubeTuneId> =
                    local_tuner!("addcmul-backward");

                let tunables = TUNER.init(move || {
                    let launch_group =
                        TuneGroup::<AddcmulBackwardAutotuneKey>::new("line_size_reduce_tile", |_| {
                            1
                        });
                    let mut set = TunableSet::new(key, input_gen);

                    for line_size in LINE_SIZE_CANDIDATES {
                        for block_size in SCALE_REDUCE_BLOCK_SIZE_CANDIDATES {
                            for bt_tile in SCALE_REDUCE_BT_TILE_CANDIDATES {
                                set = set.with(
                                    Tunable::new(
                                        format!("line_{line_size}_block_{block_size}_bt_{bt_tile}"),
                                        (move |diff: CubeTensor<R>,
                                               scale: CubeTensor<R>,
                                               output_grad: CubeTensor<R>| {
                                            let shape = diff.meta.shape().clone();
                                            let embedded_dim = shape[2];
                                            let client = diff.client.clone();
                                            let device = diff.device.clone();

                                            let base_grad = output_grad.clone();
                                            let diff_grad = empty_device::<R, F>(
                                                client.clone(),
                                                device.clone(),
                                                shape.clone(),
                                            );

                                            if shape.num_elements() > 0 {
                                                let diff_address_type = max_address_type(&[
                                                    &output_grad,
                                                    &scale,
                                                    &diff_grad,
                                                ]);
                                                let working_units =
                                                    shape.num_elements() / line_size;
                                                let cube_dim =
                                                    CubeDim::new(&client, working_units);
                                                let cube_count = calculate_cube_count_elemwise(
                                                    &client,
                                                    working_units,
                                                    cube_dim,
                                                );

                                                // SAFETY: The public addcmul input contract checks dtype/device/shape/contiguity before
                                                // primitive dispatch. Autotune only enables vector sizes that divide the embedding axis,
                                                // so the linear vector views cover exactly `num_elements / line_size` lanes.
                                                unsafe {
                                                    addcmul_backward_diff_kernel::launch_unchecked::<
                                                        F,
                                                        R,
                                                    >(
                                                        &client,
                                                        cube_count,
                                                        cube_dim,
                                                        diff_address_type,
                                                        line_size,
                                                        output_grad.clone().into_linear_view(),
                                                        scale.into_linear_view(),
                                                        diff_grad.clone().into_linear_view(),
                                                    );
                                                }
                                            }

                                            let bt_len = shape[0] * shape[1];
                                            let scale_grad_shape =
                                                burn::tensor::Shape::new([1, 1, embedded_dim]);
                                            let scale_grad = if bt_len == 0 {
                                                zeros_client::<R>(
                                                    client.clone(),
                                                    device,
                                                    scale_grad_shape,
                                                    diff.dtype,
                                                )
                                            } else {
                                                let num_bt_tiles = bt_len.div_ceil(bt_tile);
                                                let partial_shape = burn::tensor::Shape::new([
                                                    num_bt_tiles,
                                                    embedded_dim,
                                                ]);
                                                let partial_sums = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    partial_shape,
                                                );
                                                let scale_grad = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    scale_grad_shape.clone(),
                                                );
                                                let channel_vecs = embedded_dim / line_size;
                                                let cubes_x =
                                                    channel_vecs.div_ceil(block_size as usize)
                                                        as u32;
                                                let partial_address_type = max_address_type(&[
                                                    &diff,
                                                    &output_grad,
                                                    &partial_sums,
                                                ]);

                                                // Backward has two mathematical parts. `base_grad` is exactly `output_grad`, and
                                                // `diff_grad[b, t, e] = output_grad[b, t, e] * scale[0, 0, e]` stays elementwise. The
                                                // scale gradient is the only reduction:
                                                // `scale_grad[0, 0, e] = sum_{b,t} output_grad[b, t, e] * diff[b, t, e]`.
                                                // This launch tiles the combined `[batch_size, context_len]` axis and assigns parallel
                                                // embedded-dimension-vector work across `embedded_dim`. Each partial worker streams contiguous
                                                // `diff` and `output_grad` vectors for one embedded-dimension vector, keeping only an
                                                // accumulator vector live. The finalize kernel then reduces the partial sums for each
                                                // embedded-dimension vector. The design favors vectorized sequential reads along the
                                                // flattened context axis and relies on global-memory/L2 behavior for partial-sum reuse.
                                                // SAFETY: The same input contract and autotune vector-size filter make all linear vector
                                                // views valid. The partial kernel bounds-checks embedded-dimension vectors and limits each
                                                // BT tile.
                                                unsafe {
                                                    addcmul_scale_reduce_partial_kernel::launch_unchecked::<F, R>(
                                                        &client,
                                                        CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                        CubeDim::new_1d(block_size),
                                                        partial_address_type,
                                                        line_size,
                                                        diff.into_linear_view(),
                                                        output_grad.into_linear_view(),
                                                        partial_sums.clone().into_linear_view(),
                                                        channel_vecs,
                                                        bt_len,
                                                        bt_tile,
                                                    );
                                                }

                                                let finalize_address_type =
                                                    max_address_type(&[&partial_sums, &scale_grad]);
                                                // SAFETY: `partial_sums` and `scale_grad` are allocated with the launch shapes above, and
                                                // the finalize kernel guards embedded-dimension vectors against the tuned grid width.
                                                unsafe {
                                                    addcmul_scale_reduce_finalize_kernel::launch_unchecked::<F, R>(
                                                        &client,
                                                        CubeCount::Static(cubes_x, 1, 1),
                                                        CubeDim::new_1d(block_size),
                                                        finalize_address_type,
                                                        line_size,
                                                        partial_sums.into_linear_view(),
                                                        scale_grad.clone().into_linear_view(),
                                                        channel_vecs,
                                                        num_bt_tiles,
                                                    );
                                                }

                                                scale_grad
                                            };

                                            AddcmulBackwardPrimitiveOutputs::<
                                                CubeBackend<R, F, I, BT>,
                                            > {
                                                base_grad,
                                                diff_grad,
                                                scale_grad,
                                            }
                                        })
                                        .ok(),
                                    )
                                    .group(&launch_group, move |key| {
                                        if line_size <= key.max_line_size
                                            && key.embedded_dim.is_multiple_of(line_size)
                                        {
                                            1
                                        } else {
                                            -1
                                        }
                                    }),
                                );
                            }
                        }
                    }

                    set
                });

                let grads_out = TUNER.execute(
                    &CubeTuneId::new(&diff.client, &diff.device),
                    &client,
                    tunables,
                    (diff, scale, output_grad),
                );

                if let Some(node) = node_base {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.base_grad);
                }
                if let Some(node) = node_diff {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.diff_grad);
                }
                if let Some(node) = node_scale {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.scale_grad);
                }
            }
        }

        let AddcmulForwardPrimitiveInputs { base, diff, scale } = inputs;
        let output = CubeBackend::<R, F, I, BT>::fused_addcmul(AddcmulForwardPrimitiveInputs {
            base: base.primitive.clone(),
            diff: diff.primitive.clone(),
            scale: scale.primitive.clone(),
        });

        match FusedAddcmulBackward
            .prepare::<C>([base.node.clone(), diff.node.clone(), scale.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let diff_state = prep.checkpoint(&diff);
                prep.finish((diff_state, scale.primitive), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }

    fn fused_addcmul5(
        inputs: Addcmul5ForwardPrimitiveInputs<Self>,
    ) -> Addcmul5ForwardPrimitiveOutput<Self> {
        #[derive(Debug)]
        struct FusedAddcmulBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 3> for FusedAddcmulBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = (NodeId, FloatTensor<CubeBackend<R, F, I, BT>>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_base, node_diff, node_scale] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let (diff_state, scale) = ops.state;
                let diff: FloatTensor<CubeBackend<R, F, I, BT>> =
                    checkpointer.retrieve_node_output(diff_state);

                assert!(diff.is_contiguous(), "diff must be contiguous");
                assert!(scale.is_contiguous(), "scale must be contiguous");
                assert!(
                    output_grad.is_contiguous(),
                    "output_grad must be contiguous"
                );

                let client = diff.client.clone();
                let key =
                    |diff: &CubeTensor<R>, scale: &CubeTensor<R>, output_grad: &CubeTensor<R>| {
                        let shape = diff.meta.shape();
                        let max_line_size = [&diff, &scale, &output_grad]
                            .into_iter()
                            .map(|tensor| {
                                tensor_vector_size_parallel(
                                    tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                                    tensor.meta.shape(),
                                    tensor.meta.strides(),
                                    shape.num_dims() - 1,
                                )
                            })
                            .min()
                            .unwrap_or(1)
                            .max(1);

                        AddcmulBackwardAutotuneKey {
                            dtype: diff.dtype,
                            num_elements: anchor(shape.num_elements(), None, Some(1), None),
                            embedded_dim: shape[2],
                            bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                            max_line_size,
                        }
                    };

                let input_gen = |_key: &AddcmulBackwardAutotuneKey,
                                 diff: &CubeTensor<R>,
                                 scale: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                    (diff.copy(), scale.copy(), output_grad.copy())
                };

                static TUNER: LocalTuner<AddcmulBackwardAutotuneKey, CubeTuneId> =
                    local_tuner!("addcmul-backward");

                let tunables = TUNER.init(move || {
                    let launch_group =
                        TuneGroup::<AddcmulBackwardAutotuneKey>::new("line_size_reduce_tile", |_| {
                            1
                        });
                    let mut set = TunableSet::new(key, input_gen);

                    for line_size in LINE_SIZE_CANDIDATES {
                        for block_size in SCALE_REDUCE_BLOCK_SIZE_CANDIDATES {
                            for bt_tile in SCALE_REDUCE_BT_TILE_CANDIDATES {
                                set = set.with(
                                    Tunable::new(
                                        format!("line_{line_size}_block_{block_size}_bt_{bt_tile}"),
                                        (move |diff: CubeTensor<R>,
                                               scale: CubeTensor<R>,
                                               output_grad: CubeTensor<R>| {
                                            let shape = diff.meta.shape().clone();
                                            let embedded_dim = shape[2];
                                            let client = diff.client.clone();
                                            let device = diff.device.clone();

                                            let base_grad = output_grad.clone();
                                            let diff_grad = empty_device::<R, F>(
                                                client.clone(),
                                                device.clone(),
                                                shape.clone(),
                                            );

                                            if shape.num_elements() > 0 {
                                                let diff_address_type = max_address_type(&[
                                                    &output_grad,
                                                    &scale,
                                                    &diff_grad,
                                                ]);
                                                let working_units =
                                                    shape.num_elements() / line_size;
                                                let cube_dim =
                                                    CubeDim::new(&client, working_units);
                                                let cube_count = calculate_cube_count_elemwise(
                                                    &client,
                                                    working_units,
                                                    cube_dim,
                                                );

                                                // SAFETY: The public addcmul input contract checks dtype/device/shape/contiguity before
                                                // primitive dispatch. Autotune only enables vector sizes that divide the embedding axis,
                                                // so the linear vector views cover exactly `num_elements / line_size` lanes.
                                                unsafe {
                                                    addcmul_backward_diff_kernel::launch_unchecked::<
                                                        F,
                                                        R,
                                                    >(
                                                        &client,
                                                        cube_count,
                                                        cube_dim,
                                                        diff_address_type,
                                                        line_size,
                                                        output_grad.clone().into_linear_view(),
                                                        scale.into_linear_view(),
                                                        diff_grad.clone().into_linear_view(),
                                                    );
                                                }
                                            }

                                            let bt_len = shape[0] * shape[1];
                                            let scale_grad_shape =
                                                burn::tensor::Shape::new([1, 1, embedded_dim]);
                                            let scale_grad = if bt_len == 0 {
                                                zeros_client::<R>(
                                                    client.clone(),
                                                    device,
                                                    scale_grad_shape,
                                                    diff.dtype,
                                                )
                                            } else {
                                                let num_bt_tiles = bt_len.div_ceil(bt_tile);
                                                let partial_shape = burn::tensor::Shape::new([
                                                    num_bt_tiles,
                                                    embedded_dim,
                                                ]);
                                                let partial_sums = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    partial_shape,
                                                );
                                                let scale_grad = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    scale_grad_shape.clone(),
                                                );
                                                let channel_vecs = embedded_dim / line_size;
                                                let cubes_x =
                                                    channel_vecs.div_ceil(block_size as usize)
                                                        as u32;
                                                let partial_address_type = max_address_type(&[
                                                    &diff,
                                                    &output_grad,
                                                    &partial_sums,
                                                ]);

                                                // Backward has two mathematical parts. `base_grad` is exactly `output_grad`, and
                                                // `diff_grad[b, t, e] = output_grad[b, t, e] * scale[0, 0, e]` stays elementwise. The
                                                // scale gradient is the only reduction:
                                                // `scale_grad[0, 0, e] = sum_{b,t} output_grad[b, t, e] * diff[b, t, e]`.
                                                // This launch tiles the combined `[batch_size, context_len]` axis and assigns parallel
                                                // embedded-dimension-vector work across `embedded_dim`. Each partial worker streams contiguous
                                                // `diff` and `output_grad` vectors for one embedded-dimension vector, keeping only an
                                                // accumulator vector live. The finalize kernel then reduces the partial sums for each
                                                // embedded-dimension vector. The design favors vectorized sequential reads along the
                                                // flattened context axis and relies on global-memory/L2 behavior for partial-sum reuse.
                                                // SAFETY: The same input contract and autotune vector-size filter make all linear vector
                                                // views valid. The partial kernel bounds-checks embedded-dimension vectors and limits each
                                                // BT tile.
                                                unsafe {
                                                    addcmul_scale_reduce_partial_kernel::launch_unchecked::<F, R>(
                                                        &client,
                                                        CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                        CubeDim::new_1d(block_size),
                                                        partial_address_type,
                                                        line_size,
                                                        diff.into_linear_view(),
                                                        output_grad.into_linear_view(),
                                                        partial_sums.clone().into_linear_view(),
                                                        channel_vecs,
                                                        bt_len,
                                                        bt_tile,
                                                    );
                                                }

                                                let finalize_address_type =
                                                    max_address_type(&[&partial_sums, &scale_grad]);
                                                // SAFETY: `partial_sums` and `scale_grad` are allocated with the launch shapes above, and
                                                // the finalize kernel guards embedded-dimension vectors against the tuned grid width.
                                                unsafe {
                                                    addcmul_scale_reduce_finalize_kernel::launch_unchecked::<F, R>(
                                                        &client,
                                                        CubeCount::Static(cubes_x, 1, 1),
                                                        CubeDim::new_1d(block_size),
                                                        finalize_address_type,
                                                        line_size,
                                                        partial_sums.into_linear_view(),
                                                        scale_grad.clone().into_linear_view(),
                                                        channel_vecs,
                                                        num_bt_tiles,
                                                    );
                                                }

                                                scale_grad
                                            };

                                            AddcmulBackwardPrimitiveOutputs::<
                                                CubeBackend<R, F, I, BT>,
                                            > {
                                                base_grad,
                                                diff_grad,
                                                scale_grad,
                                            }
                                        })
                                        .ok(),
                                    )
                                    .group(&launch_group, move |key| {
                                        if line_size <= key.max_line_size
                                            && key.embedded_dim.is_multiple_of(line_size)
                                        {
                                            1
                                        } else {
                                            -1
                                        }
                                    }),
                                );
                            }
                        }
                    }

                    set
                });

                let grads_out = TUNER.execute(
                    &CubeTuneId::new(&diff.client, &diff.device),
                    &client,
                    tunables,
                    (diff, scale, output_grad),
                );

                if let Some(node) = node_base {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.base_grad);
                }
                if let Some(node) = node_diff {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.diff_grad);
                }
                if let Some(node) = node_scale {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.scale_grad);
                }
            }
        }

        let Addcmul5ForwardPrimitiveInputs {
            base,
            diff,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
        } = inputs;

        let output = CubeBackend::<R, F, I, BT>::fused_addcmul5(Addcmul5ForwardPrimitiveInputs {
            base: base.primitive.clone(),
            diff: diff.primitive.clone(),
            receptance_scale: receptance_scale.primitive.clone(),
            weight_decay_scale: weight_decay_scale.primitive.clone(),
            key_scale: key_scale.primitive.clone(),
            value_scale: value_scale.primitive.clone(),
            learning_rate_scale: learning_rate_scale.primitive.clone(),
        });

        let attach_backward =
            |base: FloatTensor<Autodiff<CubeBackend<R, F, I, BT>, C>>,
             diff: FloatTensor<Autodiff<CubeBackend<R, F, I, BT>, C>>,
             scale: FloatTensor<Autodiff<CubeBackend<R, F, I, BT>, C>>,
             output: FloatTensor<CubeBackend<R, F, I, BT>>| {
                match FusedAddcmulBackward
                    .prepare::<C>([base.node.clone(), diff.node.clone(), scale.node.clone()])
                    .compute_bound()
                    .stateful()
                {
                    OpsKind::Tracked(mut prep) => {
                        let diff_state = prep.checkpoint(&diff);
                        prep.finish((diff_state, scale.primitive), output)
                    }
                    OpsKind::UnTracked(prep) => prep.finish(output),
                }
            };

        Addcmul5ForwardPrimitiveOutput {
            receptance_input: attach_backward(
                base.clone(),
                diff.clone(),
                receptance_scale,
                output.receptance_input,
            ),
            weight_decay_input: attach_backward(
                base.clone(),
                diff.clone(),
                weight_decay_scale,
                output.weight_decay_input,
            ),
            key_input: attach_backward(base.clone(), diff.clone(), key_scale, output.key_input),
            value_input: attach_backward(
                base.clone(),
                diff.clone(),
                value_scale,
                output.value_input,
            ),
            learning_rate_input: attach_backward(
                base,
                diff,
                learning_rate_scale,
                output.learning_rate_input,
            ),
        }
    }
}

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const SCALE_REDUCE_BLOCK_SIZE_CANDIDATES: [u32; 3] = [64, 128, 256];
const SCALE_REDUCE_BT_TILE_CANDIDATES: [usize; 3] = [64, 128, 256];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct AddcmulBackwardAutotuneKey {
    dtype: burn::tensor::DType,
    num_elements: usize,
    embedded_dim: usize,
    bt_len: usize,
    max_line_size: usize,
}

impl core::fmt::Display for AddcmulBackwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}:{}",
            self.dtype, self.num_elements, self.embedded_dim, self.bt_len, self.max_line_size
        )
    }
}

impl AutotuneKey for AddcmulBackwardAutotuneKey {}

impl<R, F, I, BT> AutotuneOutput for AddcmulBackwardPrimitiveOutputs<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        AutotuneOutput::check_equivalence(&self.base_grad, other.base_grad);
        AutotuneOutput::check_equivalence(&self.diff_grad, other.diff_grad);
        AutotuneOutput::check_equivalence(&self.scale_grad, other.scale_grad);
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::Shape;
    use burn_fusion::{
        Fusion,
        FusionBackend,
        FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;

    impl<R, F, I, BT, C> AddcmulBackend for Autodiff<Fusion<CubeBackend<R, F, I, BT>>, C>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
        C: CheckpointStrategy,
    {
        fn fused_addcmul(inputs: AddcmulForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
            #[derive(Debug)]
            struct FusedAddcmulFusionBackward;

            impl<R, F, I, BT> Backward<Fusion<CubeBackend<R, F, I, BT>>, 3> for FusedAddcmulFusionBackward
            where
                R: CubeRuntime,
                F: FloatElement + CubeElement,
                I: IntElement,
                BT: BoolElement,
            {
                type State = (NodeId, FloatTensor<Fusion<CubeBackend<R, F, I, BT>>>);

                fn backward(
                    self,
                    ops: Ops<Self::State, 3>,
                    grads: &mut Gradients,
                    checkpointer: &mut Checkpointer,
                ) {
                    let [node_base, node_diff, node_scale] = ops.parents;
                    let output_grad = grads.consume::<Fusion<CubeBackend<R, F, I, BT>>>(&ops.node);
                    let (diff_state, scale) = ops.state;
                    let diff: FloatTensor<Fusion<CubeBackend<R, F, I, BT>>> =
                        checkpointer.retrieve_node_output(diff_state);
                    let client = diff.client.clone();
                    let [batch_size, context_len, embedded_dim] = diff.shape.dims();

                    #[derive(Clone, Debug)]
                    struct AddcmulBackwardOp<R, F, I, BT> {
                        desc: CustomOpIr,
                        _backend: core::marker::PhantomData<(R, F, I, BT)>,
                    }

                    impl<R, F, I, BT>
                        Operation<<CubeBackend<R, F, I, BT> as FusionBackend>::FusionRuntime>
                        for AddcmulBackwardOp<R, F, I, BT>
                    where
                        R: CubeRuntime,
                        F: FloatElement + CubeElement,
                        I: IntElement,
                        BT: BoolElement,
                    {
                        fn execute(
                            &self,
                            handles: &mut HandleContainer<
                                <<CubeBackend<R, F, I, BT> as FusionBackend>::FusionRuntime as FusionRuntime>::FusionHandle,
                            >,
                        ) {
                            let ([diff, scale, output_grad], [base_grad, diff_grad, scale_grad]) =
                                self.desc.as_fixed();

                            let diff = handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(diff);
                            let scale = handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(scale);
                            let output_grad =
                                handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(output_grad);

                            assert!(diff.is_contiguous(), "diff must be contiguous");
                            assert!(scale.is_contiguous(), "scale must be contiguous");
                            assert!(
                                output_grad.is_contiguous(),
                                "output_grad must be contiguous"
                            );

                            let client = diff.client.clone();
                            let key =
                                |diff: &CubeTensor<R>,
                                 scale: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                                    let shape = diff.meta.shape();
                                    let max_line_size = [&diff, &scale, &output_grad]
                                        .into_iter()
                                        .map(|tensor| {
                                            tensor_vector_size_parallel(
                                                tensor
                                                    .client
                                                    .io_optimized_vector_sizes(tensor.dtype.size()),
                                                tensor.meta.shape(),
                                                tensor.meta.strides(),
                                                shape.num_dims() - 1,
                                            )
                                        })
                                        .min()
                                        .unwrap_or(1)
                                        .max(1);

                                    AddcmulBackwardAutotuneKey {
                                        dtype: diff.dtype,
                                        num_elements: anchor(
                                            shape.num_elements(),
                                            None,
                                            Some(1),
                                            None,
                                        ),
                                        embedded_dim: shape[2],
                                        bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                                        max_line_size,
                                    }
                                };

                            let input_gen =
                                |_key: &AddcmulBackwardAutotuneKey,
                                 diff: &CubeTensor<R>,
                                 scale: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                                    (diff.copy(), scale.copy(), output_grad.copy())
                                };

                            static TUNER: LocalTuner<AddcmulBackwardAutotuneKey, CubeTuneId> =
                                local_tuner!("addcmul-fusion-backward");

                            let tunables = TUNER.init(move || {
                                let launch_group =
                                    TuneGroup::<AddcmulBackwardAutotuneKey>::new(
                                        "line_size_reduce_tile",
                                        |_| 1,
                                    );
                                let mut set = TunableSet::new(key, input_gen);

                                for line_size in LINE_SIZE_CANDIDATES {
                                    for block_size in SCALE_REDUCE_BLOCK_SIZE_CANDIDATES {
                                        for bt_tile in SCALE_REDUCE_BT_TILE_CANDIDATES {
                                            set = set.with(
                                                Tunable::new(
                                                    format!(
                                                        "line_{line_size}_block_{block_size}_bt_{bt_tile}"
                                                    ),
                                                    (move |diff: CubeTensor<R>,
                                                           scale: CubeTensor<R>,
                                                           output_grad: CubeTensor<R>| {
                                                        let shape = diff.meta.shape().clone();
                                                        let embedded_dim = shape[2];
                                                        let client = diff.client.clone();
                                                        let device = diff.device.clone();

                                                        let base_grad = output_grad.clone();
                                                        let diff_grad = empty_device::<R, F>(
                                                            client.clone(),
                                                            device.clone(),
                                                            shape.clone(),
                                                        );

                                                        if shape.num_elements() > 0 {
                                                            let diff_address_type =
                                                                max_address_type(&[
                                                                    &output_grad,
                                                                    &scale,
                                                                    &diff_grad,
                                                                ]);
                                                            let working_units =
                                                                shape.num_elements() / line_size;
                                                            let cube_dim = CubeDim::new(
                                                                &client,
                                                                working_units,
                                                            );
                                                            let cube_count =
                                                                calculate_cube_count_elemwise(
                                                                    &client,
                                                                    working_units,
                                                                    cube_dim,
                                                                );

                                                            // SAFETY: The public addcmul input contract checks dtype/device/shape/contiguity before
                                                            // primitive dispatch. Autotune only enables vector sizes that divide the embedding axis,
                                                            // so the linear vector views cover exactly `num_elements / line_size` lanes.
                                                            unsafe {
                                                                addcmul_backward_diff_kernel::launch_unchecked::<
                                                                    F,
                                                                    R,
                                                                >(
                                                                    &client,
                                                                    cube_count,
                                                                    cube_dim,
                                                                    diff_address_type,
                                                                    line_size,
                                                                    output_grad
                                                                        .clone()
                                                                        .into_linear_view(),
                                                                    scale.into_linear_view(),
                                                                    diff_grad
                                                                        .clone()
                                                                        .into_linear_view(),
                                                                );
                                                            }
                                                        }

                                                        let bt_len = shape[0] * shape[1];
                                                        let scale_grad_shape =
                                                            Shape::new([1, 1, embedded_dim]);
                                                        let scale_grad = if bt_len == 0 {
                                                            zeros_client::<R>(
                                                                client.clone(),
                                                                device,
                                                                scale_grad_shape,
                                                                diff.dtype,
                                                            )
                                                        } else {
                                                            let num_bt_tiles =
                                                                bt_len.div_ceil(bt_tile);
                                                            let partial_shape = Shape::new([
                                                                num_bt_tiles,
                                                                embedded_dim,
                                                            ]);
                                                            let partial_sums =
                                                                empty_device::<R, F>(
                                                                    client.clone(),
                                                                    device.clone(),
                                                                    partial_shape,
                                                                );
                                                            let scale_grad =
                                                                empty_device::<R, F>(
                                                                    client.clone(),
                                                                    device.clone(),
                                                                    scale_grad_shape.clone(),
                                                                );
                                                            let channel_vecs =
                                                                embedded_dim / line_size;
                                                            let cubes_x = channel_vecs
                                                                .div_ceil(block_size as usize)
                                                                as u32;
                                                            let partial_address_type =
                                                                max_address_type(&[
                                                                    &diff,
                                                                    &output_grad,
                                                                    &partial_sums,
                                                                ]);

                                                            // Backward has two mathematical parts. `base_grad` is exactly `output_grad`, and
                                                            // `diff_grad[b, t, e] = output_grad[b, t, e] * scale[0, 0, e]` stays elementwise. The
                                                            // scale gradient is the only reduction:
                                                            // `scale_grad[0, 0, e] = sum_{b,t} output_grad[b, t, e] * diff[b, t, e]`.
                                                            // This launch tiles the combined `[batch_size, context_len]` axis and assigns parallel
                                                            // embedded-dimension-vector work across `embedded_dim`. Each partial worker streams contiguous
                                                            // `diff` and `output_grad` vectors for one embedded-dimension vector, keeping only an
                                                            // accumulator vector live. The finalize kernel then reduces the partial sums for each
                                                            // embedded-dimension vector. The design favors vectorized sequential reads along the
                                                            // flattened context axis and relies on global-memory/L2 behavior for partial-sum reuse.
                                                            // SAFETY: The same input contract and autotune vector-size filter make all linear vector
                                                            // views valid. The partial kernel bounds-checks embedded-dimension vectors and limits each
                                                            // BT tile.
                                                            unsafe {
                                                                addcmul_scale_reduce_partial_kernel::launch_unchecked::<F, R>(
                                                                    &client,
                                                                    CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                                    CubeDim::new_1d(block_size),
                                                                    partial_address_type,
                                                                    line_size,
                                                                    diff.into_linear_view(),
                                                                    output_grad.into_linear_view(),
                                                                    partial_sums.clone().into_linear_view(),
                                                                    channel_vecs,
                                                                    bt_len,
                                                                    bt_tile,
                                                                );
                                                            }

                                                            let finalize_address_type =
                                                                max_address_type(&[
                                                                    &partial_sums,
                                                                    &scale_grad,
                                                                ]);
                                                            // SAFETY: `partial_sums` and `scale_grad` are allocated with the launch shapes above, and
                                                            // the finalize kernel guards embedded-dimension vectors against the tuned grid width.
                                                            unsafe {
                                                                addcmul_scale_reduce_finalize_kernel::launch_unchecked::<F, R>(
                                                                    &client,
                                                                    CubeCount::Static(cubes_x, 1, 1),
                                                                    CubeDim::new_1d(block_size),
                                                                    finalize_address_type,
                                                                    line_size,
                                                                    partial_sums.into_linear_view(),
                                                                    scale_grad.clone().into_linear_view(),
                                                                    channel_vecs,
                                                                    num_bt_tiles,
                                                                );
                                                            }

                                                            scale_grad
                                                        };

                                                        AddcmulBackwardPrimitiveOutputs::<
                                                            CubeBackend<R, F, I, BT>,
                                                        > {
                                                            base_grad,
                                                            diff_grad,
                                                            scale_grad,
                                                        }
                                                    })
                                                    .ok(),
                                                )
                                                .group(&launch_group, move |key| {
                                                    if line_size <= key.max_line_size
                                                        && key
                                                            .embedded_dim
                                                            .is_multiple_of(line_size)
                                                    {
                                                        1
                                                    } else {
                                                        -1
                                                    }
                                                }),
                                            );
                                        }
                                    }
                                }

                                set
                            });

                            let grads_out = TUNER.execute(
                                &CubeTuneId::new(&diff.client, &diff.device),
                                &client,
                                tunables,
                                (diff, scale, output_grad),
                            );

                            handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                                &base_grad.id,
                                grads_out.base_grad,
                            );
                            handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                                &diff_grad.id,
                                grads_out.diff_grad,
                            );
                            handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                                &scale_grad.id,
                                grads_out.scale_grad,
                            );
                        }
                    }

                    let mut streams = OperationStreams::default();
                    streams.tensor(&diff);
                    streams.tensor(&scale);
                    streams.tensor(&output_grad);

                    let output_desc = [
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size, context_len, embedded_dim]),
                            F::dtype(),
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size, context_len, embedded_dim]),
                            F::dtype(),
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([1, 1, embedded_dim]),
                            F::dtype(),
                        ),
                    ];

                    let desc = CustomOpIr::new(
                        "fused_addcmul_backward",
                        &[diff.into_ir(), scale.into_ir(), output_grad.into_ir()],
                        &output_desc,
                    );
                    let op = AddcmulBackwardOp::<R, F, I, BT> {
                        desc,
                        _backend: core::marker::PhantomData,
                    };
                    let mut outputs =
                        client.register(streams, OperationIr::Custom(op.desc.clone()), op);
                    let scale_grad = outputs.pop().expect("missing scale_grad");
                    let diff_grad = outputs.pop().expect("missing diff_grad");
                    let base_grad = outputs.pop().expect("missing base_grad");

                    if let Some(node) = node_base {
                        grads.register::<Fusion<CubeBackend<R, F, I, BT>>>(node.id, base_grad);
                    }
                    if let Some(node) = node_diff {
                        grads.register::<Fusion<CubeBackend<R, F, I, BT>>>(node.id, diff_grad);
                    }
                    if let Some(node) = node_scale {
                        grads.register::<Fusion<CubeBackend<R, F, I, BT>>>(node.id, scale_grad);
                    }
                }
            }

            let AddcmulForwardPrimitiveInputs { base, diff, scale } = inputs;
            let output =
                Fusion::<CubeBackend<R, F, I, BT>>::fused_addcmul(AddcmulForwardPrimitiveInputs {
                    base: base.primitive.clone(),
                    diff: diff.primitive.clone(),
                    scale: scale.primitive.clone(),
                });

            match FusedAddcmulFusionBackward
                .prepare::<C>([base.node.clone(), diff.node.clone(), scale.node.clone()])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(mut prep) => {
                    let diff_state = prep.checkpoint(&diff);
                    prep.finish((diff_state, scale.primitive), output)
                }
                OpsKind::UnTracked(prep) => prep.finish(output),
            }
        }

        fn fused_addcmul5(
            inputs: Addcmul5ForwardPrimitiveInputs<Self>,
        ) -> Addcmul5ForwardPrimitiveOutput<Self> {
            #[derive(Debug)]
            struct FusedAddcmulFusionBackward;

            impl<R, F, I, BT> Backward<Fusion<CubeBackend<R, F, I, BT>>, 3> for FusedAddcmulFusionBackward
            where
                R: CubeRuntime,
                F: FloatElement + CubeElement,
                I: IntElement,
                BT: BoolElement,
            {
                type State = (NodeId, FloatTensor<Fusion<CubeBackend<R, F, I, BT>>>);

                fn backward(
                    self,
                    ops: Ops<Self::State, 3>,
                    grads: &mut Gradients,
                    checkpointer: &mut Checkpointer,
                ) {
                    let [node_base, node_diff, node_scale] = ops.parents;
                    let output_grad = grads.consume::<Fusion<CubeBackend<R, F, I, BT>>>(&ops.node);
                    let (diff_state, scale) = ops.state;
                    let diff: FloatTensor<Fusion<CubeBackend<R, F, I, BT>>> =
                        checkpointer.retrieve_node_output(diff_state);
                    let client = diff.client.clone();
                    let [batch_size, context_len, embedded_dim] = diff.shape.dims();

                    #[derive(Clone, Debug)]
                    struct AddcmulBackwardOp<R, F, I, BT> {
                        desc: CustomOpIr,
                        _backend: core::marker::PhantomData<(R, F, I, BT)>,
                    }

                    impl<R, F, I, BT>
                        Operation<<CubeBackend<R, F, I, BT> as FusionBackend>::FusionRuntime>
                        for AddcmulBackwardOp<R, F, I, BT>
                    where
                        R: CubeRuntime,
                        F: FloatElement + CubeElement,
                        I: IntElement,
                        BT: BoolElement,
                    {
                        fn execute(
                            &self,
                            handles: &mut HandleContainer<
                                <<CubeBackend<R, F, I, BT> as FusionBackend>::FusionRuntime as FusionRuntime>::FusionHandle,
                            >,
                        ) {
                            let ([diff, scale, output_grad], [base_grad, diff_grad, scale_grad]) =
                                self.desc.as_fixed();

                            let diff = handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(diff);
                            let scale = handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(scale);
                            let output_grad =
                                handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(output_grad);

                            assert!(diff.is_contiguous(), "diff must be contiguous");
                            assert!(scale.is_contiguous(), "scale must be contiguous");
                            assert!(
                                output_grad.is_contiguous(),
                                "output_grad must be contiguous"
                            );

                            let client = diff.client.clone();
                            let key =
                                |diff: &CubeTensor<R>,
                                 scale: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                                    let shape = diff.meta.shape();
                                    let max_line_size = [&diff, &scale, &output_grad]
                                        .into_iter()
                                        .map(|tensor| {
                                            tensor_vector_size_parallel(
                                                tensor
                                                    .client
                                                    .io_optimized_vector_sizes(tensor.dtype.size()),
                                                tensor.meta.shape(),
                                                tensor.meta.strides(),
                                                shape.num_dims() - 1,
                                            )
                                        })
                                        .min()
                                        .unwrap_or(1)
                                        .max(1);

                                    AddcmulBackwardAutotuneKey {
                                        dtype: diff.dtype,
                                        num_elements: anchor(
                                            shape.num_elements(),
                                            None,
                                            Some(1),
                                            None,
                                        ),
                                        embedded_dim: shape[2],
                                        bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                                        max_line_size,
                                    }
                                };

                            let input_gen =
                                |_key: &AddcmulBackwardAutotuneKey,
                                 diff: &CubeTensor<R>,
                                 scale: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                                    (diff.copy(), scale.copy(), output_grad.copy())
                                };

                            static TUNER: LocalTuner<AddcmulBackwardAutotuneKey, CubeTuneId> =
                                local_tuner!("addcmul-fusion-backward");

                            let tunables = TUNER.init(move || {
                                let launch_group =
                                    TuneGroup::<AddcmulBackwardAutotuneKey>::new(
                                        "line_size_reduce_tile",
                                        |_| 1,
                                    );
                                let mut set = TunableSet::new(key, input_gen);

                                for line_size in LINE_SIZE_CANDIDATES {
                                    for block_size in SCALE_REDUCE_BLOCK_SIZE_CANDIDATES {
                                        for bt_tile in SCALE_REDUCE_BT_TILE_CANDIDATES {
                                            set = set.with(
                                                Tunable::new(
                                                    format!(
                                                        "line_{line_size}_block_{block_size}_bt_{bt_tile}"
                                                    ),
                                                    (move |diff: CubeTensor<R>,
                                                           scale: CubeTensor<R>,
                                                           output_grad: CubeTensor<R>| {
                                                        let shape = diff.meta.shape().clone();
                                                        let embedded_dim = shape[2];
                                                        let client = diff.client.clone();
                                                        let device = diff.device.clone();

                                                        let base_grad = output_grad.clone();
                                                        let diff_grad = empty_device::<R, F>(
                                                            client.clone(),
                                                            device.clone(),
                                                            shape.clone(),
                                                        );

                                                        if shape.num_elements() > 0 {
                                                            let diff_address_type =
                                                                max_address_type(&[
                                                                    &output_grad,
                                                                    &scale,
                                                                    &diff_grad,
                                                                ]);
                                                            let working_units =
                                                                shape.num_elements() / line_size;
                                                            let cube_dim = CubeDim::new(
                                                                &client,
                                                                working_units,
                                                            );
                                                            let cube_count =
                                                                calculate_cube_count_elemwise(
                                                                    &client,
                                                                    working_units,
                                                                    cube_dim,
                                                                );

                                                            // SAFETY: The public addcmul input contract checks dtype/device/shape/contiguity before
                                                            // primitive dispatch. Autotune only enables vector sizes that divide the embedding axis,
                                                            // so the linear vector views cover exactly `num_elements / line_size` lanes.
                                                            unsafe {
                                                                addcmul_backward_diff_kernel::launch_unchecked::<
                                                                    F,
                                                                    R,
                                                                >(
                                                                    &client,
                                                                    cube_count,
                                                                    cube_dim,
                                                                    diff_address_type,
                                                                    line_size,
                                                                    output_grad
                                                                        .clone()
                                                                        .into_linear_view(),
                                                                    scale.into_linear_view(),
                                                                    diff_grad
                                                                        .clone()
                                                                        .into_linear_view(),
                                                                );
                                                            }
                                                        }

                                                        let bt_len = shape[0] * shape[1];
                                                        let scale_grad_shape =
                                                            Shape::new([1, 1, embedded_dim]);
                                                        let scale_grad = if bt_len == 0 {
                                                            zeros_client::<R>(
                                                                client.clone(),
                                                                device,
                                                                scale_grad_shape,
                                                                diff.dtype,
                                                            )
                                                        } else {
                                                            let num_bt_tiles =
                                                                bt_len.div_ceil(bt_tile);
                                                            let partial_shape = Shape::new([
                                                                num_bt_tiles,
                                                                embedded_dim,
                                                            ]);
                                                            let partial_sums =
                                                                empty_device::<R, F>(
                                                                    client.clone(),
                                                                    device.clone(),
                                                                    partial_shape,
                                                                );
                                                            let scale_grad =
                                                                empty_device::<R, F>(
                                                                    client.clone(),
                                                                    device.clone(),
                                                                    scale_grad_shape.clone(),
                                                                );
                                                            let channel_vecs =
                                                                embedded_dim / line_size;
                                                            let cubes_x = channel_vecs
                                                                .div_ceil(block_size as usize)
                                                                as u32;
                                                            let partial_address_type =
                                                                max_address_type(&[
                                                                    &diff,
                                                                    &output_grad,
                                                                    &partial_sums,
                                                                ]);

                                                            // Backward has two mathematical parts. `base_grad` is exactly `output_grad`, and
                                                            // `diff_grad[b, t, e] = output_grad[b, t, e] * scale[0, 0, e]` stays elementwise. The
                                                            // scale gradient is the only reduction:
                                                            // `scale_grad[0, 0, e] = sum_{b,t} output_grad[b, t, e] * diff[b, t, e]`.
                                                            // This launch tiles the combined `[batch_size, context_len]` axis and assigns parallel
                                                            // embedded-dimension-vector work across `embedded_dim`. Each partial worker streams contiguous
                                                            // `diff` and `output_grad` vectors for one embedded-dimension vector, keeping only an
                                                            // accumulator vector live. The finalize kernel then reduces the partial sums for each
                                                            // embedded-dimension vector. The design favors vectorized sequential reads along the
                                                            // flattened context axis and relies on global-memory/L2 behavior for partial-sum reuse.
                                                            // SAFETY: The same input contract and autotune vector-size filter make all linear vector
                                                            // views valid. The partial kernel bounds-checks embedded-dimension vectors and limits each
                                                            // BT tile.
                                                            unsafe {
                                                                addcmul_scale_reduce_partial_kernel::launch_unchecked::<F, R>(
                                                                    &client,
                                                                    CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                                    CubeDim::new_1d(block_size),
                                                                    partial_address_type,
                                                                    line_size,
                                                                    diff.into_linear_view(),
                                                                    output_grad.into_linear_view(),
                                                                    partial_sums.clone().into_linear_view(),
                                                                    channel_vecs,
                                                                    bt_len,
                                                                    bt_tile,
                                                                );
                                                            }

                                                            let finalize_address_type =
                                                                max_address_type(&[
                                                                    &partial_sums,
                                                                    &scale_grad,
                                                                ]);
                                                            // SAFETY: `partial_sums` and `scale_grad` are allocated with the launch shapes above, and
                                                            // the finalize kernel guards embedded-dimension vectors against the tuned grid width.
                                                            unsafe {
                                                                addcmul_scale_reduce_finalize_kernel::launch_unchecked::<F, R>(
                                                                    &client,
                                                                    CubeCount::Static(cubes_x, 1, 1),
                                                                    CubeDim::new_1d(block_size),
                                                                    finalize_address_type,
                                                                    line_size,
                                                                    partial_sums.into_linear_view(),
                                                                    scale_grad.clone().into_linear_view(),
                                                                    channel_vecs,
                                                                    num_bt_tiles,
                                                                );
                                                            }

                                                            scale_grad
                                                        };

                                                        AddcmulBackwardPrimitiveOutputs::<
                                                            CubeBackend<R, F, I, BT>,
                                                        > {
                                                            base_grad,
                                                            diff_grad,
                                                            scale_grad,
                                                        }
                                                    })
                                                    .ok(),
                                                )
                                                .group(&launch_group, move |key| {
                                                    if line_size <= key.max_line_size
                                                        && key
                                                            .embedded_dim
                                                            .is_multiple_of(line_size)
                                                    {
                                                        1
                                                    } else {
                                                        -1
                                                    }
                                                }),
                                            );
                                        }
                                    }
                                }

                                set
                            });

                            let grads_out = TUNER.execute(
                                &CubeTuneId::new(&diff.client, &diff.device),
                                &client,
                                tunables,
                                (diff, scale, output_grad),
                            );

                            handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                                &base_grad.id,
                                grads_out.base_grad,
                            );
                            handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                                &diff_grad.id,
                                grads_out.diff_grad,
                            );
                            handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                                &scale_grad.id,
                                grads_out.scale_grad,
                            );
                        }
                    }

                    let mut streams = OperationStreams::default();
                    streams.tensor(&diff);
                    streams.tensor(&scale);
                    streams.tensor(&output_grad);

                    let output_desc = [
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size, context_len, embedded_dim]),
                            F::dtype(),
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size, context_len, embedded_dim]),
                            F::dtype(),
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([1, 1, embedded_dim]),
                            F::dtype(),
                        ),
                    ];

                    let desc = CustomOpIr::new(
                        "fused_addcmul_backward",
                        &[diff.into_ir(), scale.into_ir(), output_grad.into_ir()],
                        &output_desc,
                    );
                    let op = AddcmulBackwardOp::<R, F, I, BT> {
                        desc,
                        _backend: core::marker::PhantomData,
                    };
                    let mut outputs =
                        client.register(streams, OperationIr::Custom(op.desc.clone()), op);
                    let scale_grad = outputs.pop().expect("missing scale_grad");
                    let diff_grad = outputs.pop().expect("missing diff_grad");
                    let base_grad = outputs.pop().expect("missing base_grad");

                    if let Some(node) = node_base {
                        grads.register::<Fusion<CubeBackend<R, F, I, BT>>>(node.id, base_grad);
                    }
                    if let Some(node) = node_diff {
                        grads.register::<Fusion<CubeBackend<R, F, I, BT>>>(node.id, diff_grad);
                    }
                    if let Some(node) = node_scale {
                        grads.register::<Fusion<CubeBackend<R, F, I, BT>>>(node.id, scale_grad);
                    }
                }
            }

            let Addcmul5ForwardPrimitiveInputs {
                base,
                diff,
                receptance_scale,
                weight_decay_scale,
                key_scale,
                value_scale,
                learning_rate_scale,
            } = inputs;

            let output = Fusion::<CubeBackend<R, F, I, BT>>::fused_addcmul5(
                Addcmul5ForwardPrimitiveInputs {
                    base: base.primitive.clone(),
                    diff: diff.primitive.clone(),
                    receptance_scale: receptance_scale.primitive.clone(),
                    weight_decay_scale: weight_decay_scale.primitive.clone(),
                    key_scale: key_scale.primitive.clone(),
                    value_scale: value_scale.primitive.clone(),
                    learning_rate_scale: learning_rate_scale.primitive.clone(),
                },
            );

            let attach_backward =
                |base: FloatTensor<Autodiff<Fusion<CubeBackend<R, F, I, BT>>, C>>,
                 diff: FloatTensor<Autodiff<Fusion<CubeBackend<R, F, I, BT>>, C>>,
                 scale: FloatTensor<Autodiff<Fusion<CubeBackend<R, F, I, BT>>, C>>,
                 output: FloatTensor<Fusion<CubeBackend<R, F, I, BT>>>| {
                    match FusedAddcmulFusionBackward
                        .prepare::<C>([base.node.clone(), diff.node.clone(), scale.node.clone()])
                        .compute_bound()
                        .stateful()
                    {
                        OpsKind::Tracked(mut prep) => {
                            let diff_state = prep.checkpoint(&diff);
                            prep.finish((diff_state, scale.primitive), output)
                        }
                        OpsKind::UnTracked(prep) => prep.finish(output),
                    }
                };

            Addcmul5ForwardPrimitiveOutput {
                receptance_input: attach_backward(
                    base.clone(),
                    diff.clone(),
                    receptance_scale,
                    output.receptance_input,
                ),
                weight_decay_input: attach_backward(
                    base.clone(),
                    diff.clone(),
                    weight_decay_scale,
                    output.weight_decay_input,
                ),
                key_input: attach_backward(base.clone(), diff.clone(), key_scale, output.key_input),
                value_input: attach_backward(
                    base.clone(),
                    diff.clone(),
                    value_scale,
                    output.value_input,
                ),
                learning_rate_input: attach_backward(
                    base,
                    diff,
                    learning_rate_scale,
                    output.learning_rate_input,
                ),
            }
        }
    }
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
