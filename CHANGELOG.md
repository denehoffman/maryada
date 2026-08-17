# Changelog

## [0.2.1](https://github.com/denehoffman/maryada/compare/v0.2.0...v0.2.1) (2026-08-17)


### Features

* Add branch and bound and some complex matrix solves ([9f3deeb](https://github.com/denehoffman/maryada/commit/9f3deeb6670a65960874398523ec61bea7e7bcd4))
* **linalg:** Add diagnostic errors to interval solvers ([1f6677f](https://github.com/denehoffman/maryada/commit/1f6677fe49ad17532d544f58c6a3630a22e7c9c8))
* Make BnB easier to use ([03d71da](https://github.com/denehoffman/maryada/commit/03d71daf569136ef80e57846e7ac0004d9f9b1c3))


### Bug fixes

* Clippy lints ([d22f9db](https://github.com/denehoffman/maryada/commit/d22f9db3de317f34d75808fc03bafd6f852afc12))


### Code refactoring

* Rework trait interface ([2810880](https://github.com/denehoffman/maryada/commit/28108801d912f644231066adf0ec607703c2253b))

## [0.2.0](https://github.com/denehoffman/maryada/compare/v0.1.2...v0.2.0) (2026-08-16)


### ⚠ BREAKING CHANGES

* unify interval operations and operator overloads

### Features

* Add full set of operator overloads for interval matrices ([66cf19f](https://github.com/denehoffman/maryada/commit/66cf19fe89f95c78d1518470cad20a0387d86af2))
* Add Gauss-Seidel and reorganize module structure ([2990d1d](https://github.com/denehoffman/maryada/commit/2990d1dd954ec237c008b24bd5b196387587f241))
* Add some nice traits to help with linear algebra later (to avoid writing a bunch of implementations over capabilities) ([dccf132](https://github.com/denehoffman/maryada/commit/dccf132b0b59090e1f6a4cc225612425750177f3))
* Credit Horacek's thesis and remove linalg and complex from default features ([305a8cd](https://github.com/denehoffman/maryada/commit/305a8cdef19ce70c5e4b0d2b97927f49bbfd4c0b))
* **linalg:** Add configurable interval linear solvers ([bef764c](https://github.com/denehoffman/maryada/commit/bef764c2328910e19af5335824a9cdfeb73aab43))
* **linalg:** Add interval matrix constructors and row vectors ([227f452](https://github.com/denehoffman/maryada/commit/227f4520f005d2a52290bc337c380c31ceaf871e))
* **linalg:** Add matrix predicates and structural operations ([b5ee53e](https://github.com/denehoffman/maryada/commit/b5ee53ea01dcd106ac4eca2dec7ce97106b30fe7))
* **linalg:** Add verified interval system solvers ([17ab226](https://github.com/denehoffman/maryada/commit/17ab226851cfcfc9eb58ab2d5fa533d8dc4e3ee4))
* **linalg:** Expose matrix accessors iterators and indexing ([85eaabf](https://github.com/denehoffman/maryada/commit/85eaabf2654db909fc49ce9ec93f8da2ff119adc))
* **linalg:** Support generic interval matrix storage ([0311c9a](https://github.com/denehoffman/maryada/commit/0311c9abe1f17b032b229a1a3f28c37657d6953b))
* Update default solve/inverse method to use automatic preconditioning and gaussian elimination ([206ad4e](https://github.com/denehoffman/maryada/commit/206ad4e31ced220dd10378cacfef12a888868cd8))


### Bug fixes

* Clippy lints ([4279b3b](https://github.com/denehoffman/maryada/commit/4279b3b51bcbf735bc23698e724b72dfc5bde55c))
* **rounding:** Tighten transcendental interval enclosures ([c0a4570](https://github.com/denehoffman/maryada/commit/c0a4570fec0fce6120d75ae52586650d6f1e49e1))


### Code refactoring

* **linalg:** Modernize interval matrix operations and imports ([6f815c4](https://github.com/denehoffman/maryada/commit/6f815c43f708c6ee1b6970f4230149aff97d4987))
* Unify interval operations and operator overloads ([e2d5505](https://github.com/denehoffman/maryada/commit/e2d5505e89218791f3a2d5db5596d47654ac7e82))

## [0.1.2](https://github.com/denehoffman/maryada/compare/v0.1.1...v0.1.2) (2026-08-03)


### Features

* Expand owned interval arithmetic APIs ([2551133](https://github.com/denehoffman/maryada/commit/2551133af5ba7b31764bc115e54427db43e7de5d))

## [0.1.1](https://github.com/denehoffman/maryada/compare/v0.1.0...v0.1.1) (2026-08-03)


### Bug fixes

* Conform real interval operations to IEEE 1788.1 ([0b404b3](https://github.com/denehoffman/maryada/commit/0b404b3bb1f904607b1893df5de5b9ae3508ef76))


### Documentation

* Add badges to README.md ([0fa9f45](https://github.com/denehoffman/maryada/commit/0fa9f45155b79bbe78b247adf4df8313aa6aa044))
* Add newline after badges ([7d1f396](https://github.com/denehoffman/maryada/commit/7d1f3962e790517b9cc9a63df1160fd63c23e2ac))
* Document conformance and fix badge links ([74fb695](https://github.com/denehoffman/maryada/commit/74fb695bfd362035081b8cbc8bdc5117e08b6cf3))

## 0.1.0 (2026-08-02)


### ⚠ BREAKING CHANGES

* remove tuple-based interval and ComplexBox conversions, remove Ord from interval types, and make decorated equality structural.

### Features

* Add complex interval boxes ([aca86cf](https://github.com/denehoffman/maryada/commit/aca86cf528aef4f4fdee3d9830c6923caa4a47be))
* Add nicer chaining UX for all operations ([a16e031](https://github.com/denehoffman/maryada/commit/a16e031bbf2a5414a30d146266d25df5640596fa))
* Handle text parsing and output ([ff9d20b](https://github.com/denehoffman/maryada/commit/ff9d20bcb484402afbc2abef0a6edc4996bd25d6))
* Implement interval numeric operations ([072277d](https://github.com/denehoffman/maryada/commit/072277ddbf85443fc3b92cb16a4092f06efd6546))
* Write skeleton and some of the important interval arithmetic ([85a3cbf](https://github.com/denehoffman/maryada/commit/85a3cbffd4ed1e49c545b0379d1f2d71f2b309f6))


### Bug fixes

* Improve IEEE 1788 conformance ([0474cd9](https://github.com/denehoffman/maryada/commit/0474cd959160d903f9ec94c8e60de24d7947244e))


### Documentation

* Add crates.io package metadata ([bccd149](https://github.com/denehoffman/maryada/commit/bccd149cab45b83f7e21b5f6c2dea5364f3e5ad0))
* Add dual-license texts ([309a6b2](https://github.com/denehoffman/maryada/commit/309a6b2cc6e8cd134f6872af506e23a3d62e44e5))
* Add pi example ([0d3b9c7](https://github.com/denehoffman/maryada/commit/0d3b9c7e90c21facf636db3f2178e5b5d5d82eab))
* Document complex interval API ([f7a68b5](https://github.com/denehoffman/maryada/commit/f7a68b57df500f50b4d2e20320a6d733202428a1))
* Document public interval API ([b7f5714](https://github.com/denehoffman/maryada/commit/b7f5714a29938fb9192823cfb193e06ac8cc08c9))
* Use lowercase maryada name ([a6cbd35](https://github.com/denehoffman/maryada/commit/a6cbd35aba174ae183047ba4d96360a36bc91df3))


### Continuous integration

* Add validation and release workflows ([237dcec](https://github.com/denehoffman/maryada/commit/237dcecb9bd4d2b6b9452d5bc731cdd0435735cd))
* Configure pre-1.0 release versioning ([25151cb](https://github.com/denehoffman/maryada/commit/25151cb3e3dcc85a787ed961fb8968528decd245))
* Update release-please token secret ([bf46e8f](https://github.com/denehoffman/maryada/commit/bf46e8f21dfa920c4e55d851798f5d036b1641b4))
