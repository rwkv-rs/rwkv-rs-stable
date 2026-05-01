#[test]
fn ui() {
    // `ui` means user interface: the Rust/trybuild convention for tests that
    // check compiler diagnostics shown to users.
    let tests = trybuild::TestCases::new();

    tests.pass("tests/ui/pass/*.rs");
    tests.compile_fail("tests/ui/fail/*.rs");
}
