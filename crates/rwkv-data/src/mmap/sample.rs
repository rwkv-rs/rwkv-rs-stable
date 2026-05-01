//! Deterministic sample scheduling utilities for mmap datasets.

/// Deterministic offset sampler for distributed dataset traversal.
pub struct Sampler {
    /// Number of devices participating in the traversal.
    pub num_devices: u64,
    /// Zero-based index of this device.
    pub device_index: u64,
    /// Number of samples assigned per mini-epoch.
    pub samples_per_epoch: u64,
    /// Prime modulus used to permute sample offsets.
    pub magic_prime: u64,
}

impl Sampler {
    /// Creates a sampler with the provided distributed traversal parameters.
    pub fn new(
        num_devices: u64,
        device_index: u64,
        samples_per_epoch: u64,
        magic_prime: u64,
    ) -> Self {
        Self {
            num_devices,
            device_index,
            samples_per_epoch,
            magic_prime,
        }
    }

    /// Computes the base sample offset for an item in a mini-epoch.
    ///
    /// # Panics
    ///
    /// Panics if `magic_prime` is zero because modulo by zero is undefined.
    /// In debug builds, arithmetic overflow in the offset calculation may also
    /// panic.
    pub fn get_base_offset(&self, index: u64, mini_epoch_index: u64) -> u64 {
        let unique_sample_index = 1
            + mini_epoch_index * self.samples_per_epoch
            + (index * self.num_devices)
            + self.device_index;

        let u_mod_p = unique_sample_index % self.magic_prime;

        let u2_mod_p = (u_mod_p * u_mod_p) % self.magic_prime;

        let u3_mod_p = (u2_mod_p * u_mod_p) % self.magic_prime;

        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;

        (u3_mod_p * ((self.magic_prime as f64) * phi).floor() as u64) % self.magic_prime
    }
}

/// Finds the largest prime below `num_slots` that is congruent to 2 modulo 3.
///
/// Returns `0` when no such prime exists.
pub fn calculate_magic_prime(num_slots: u64) -> u64 {
    (0..num_slots)
        .rev()
        .find(|&x| x % 3 == 2 && primes::is_prime(x))
        .unwrap_or(0)
}
