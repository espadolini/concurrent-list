# `concurrent-list`

An append-only, single-writer multi-reader unrolled linked list.

The MSRV (Minimum Supported Rust Version) is 1.56, for [`UnsafeCell::raw_get`](https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html#method.raw_get).

`concurrent-list` is `no_std`-compatible (requiring `alloc`).

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
