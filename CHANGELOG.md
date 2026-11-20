# Changelog

## [0.3.0](https://github.com/denehoffman/maryada/compare/v0.2.1...v0.3.0) (2026-11-20)


### ⚠ BREAKING CHANGES

* **branch:** Rename gap_tolerance and its builders to use absolute_gap_tolerance. GlobalMinimizerOptions now includes relative_gap_tolerance, which defaults to 1e-6 and can be disabled independently.
* **branch:** BranchAndBound::step requires a Clone domain so it can restore the domain after a failed contraction.

### Features

* **branch:** Enable absolute and relative gap stopping ([73dd048](https://github.com/denehoffman/maryada/commit/73dd04807422954752ad7584b18899427598f7ed))
* **branch:** Expose the best evaluated singleton sample ([56e35ba](https://github.com/denehoffman/maryada/commit/56e35badafd198d1422304cb865faeb8cf95403d))
* **branch:** Isolate scalar roots with interval Newton steps ([98e9926](https://github.com/denehoffman/maryada/commit/98e99262af73029b7a211596132e6d307400d033))
* **interval:** Add checked scalar constructors ([cad32ec](https://github.com/denehoffman/maryada/commit/cad32ec5811eadf93ed7f6b846aeec021d5510f3))
* **interval:** Add directed log1p and expm1 ([f3708cd](https://github.com/denehoffman/maryada/commit/f3708cd6d8345a50c29f6596d16655b9bcb466e2))
* **interval:** Add scalar automatic differentiation ([5a4a39b](https://github.com/denehoffman/maryada/commit/5a4a39bc8b5bcef3697751c03019e9879cdafa76))
* **interval:** Add tight mathematical constant enclosures ([ae3328b](https://github.com/denehoffman/maryada/commit/ae3328bee0ffac3ff6e5232b0bd65fc63bae1bda))
* **interval:** Preserve split division and reciprocal results ([f4e50ce](https://github.com/denehoffman/maryada/commit/f4e50ce767164eeed7a9360c5afdf2558a4d00b3))
* **linalg:** Add certified Krawczyk contraction for nonlinear systems ([4f2dbb7](https://github.com/denehoffman/maryada/commit/4f2dbb749be15aa1794061ae4d28ac208d761281))
* **linalg:** Check runtime matrix operation dimensions ([6fe8d8e](https://github.com/denehoffman/maryada/commit/6fe8d8eec9ab905a8bdf43e92dbb95d5d7a99f4e))
* **serde:** Serialize intervals via interchange formats ([27c42f3](https://github.com/denehoffman/maryada/commit/27c42f31c7410045d457fbcc30942e1dc42aac61))
* **text:** Add checked parsing with typed diagnostics ([3dcafbf](https://github.com/denehoffman/maryada/commit/3dcafbf787d703e41e17d77a60864dfb8b59ad06))
* **text:** Add outward-rounded decimal interval formatting ([7265c7e](https://github.com/denehoffman/maryada/commit/7265c7e8b44bc0558be3e70f91955a4782f1b3f1))


### Bug fixes

* **branch:** Bisect the widest splittable coordinate ([337b0a1](https://github.com/denehoffman/maryada/commit/337b0a12a68a17d1beed816fa38440b1307d714d))
* **branch:** Preserve pending domains after callback errors ([ea7bc3a](https://github.com/denehoffman/maryada/commit/ea7bc3a788e2fe341112215e1535740609853e2b))
* **branch:** Round the optimality gap upward ([ad3cd25](https://github.com/denehoffman/maryada/commit/ad3cd258d5c93dfb6932d90a993b463c2f53cc17))
* **complex:** Include zero in square root ([cab704e](https://github.com/denehoffman/maryada/commit/cab704ec3651e682c2ee3d2d4c709393adb9adab))
* **complex:** Scale division and reciprocal across binary64 ranges ([8b940cf](https://github.com/denehoffman/maryada/commit/8b940cf0f2cf5db9fee63a22e5f7e817efabbb00))
* **docs:** Make the README example compile ([afb4317](https://github.com/denehoffman/maryada/commit/afb4317b527c755c02fdf2c3c3e918d506326585))
* **interval:** Compare cancellation widths exactly ([e5df0db](https://github.com/denehoffman/maryada/commit/e5df0db51d7344fde254bd198025c8dba24f9308))
* **interval:** Compute tight hypot bounds across binary64 scales ([b6fe695](https://github.com/denehoffman/maryada/commit/b6fe695bfe62577bbd88f3a297bc900fa2b606e1))
* **interval:** Preserve empty intervals during promotion ([2bb26d3](https://github.com/denehoffman/maryada/commit/2bb26d3e167827b6a529753f932f6fb62d389cd2))
* **interval:** Tighten cancellation-prone hyperbolic enclosures ([5a900fa](https://github.com/denehoffman/maryada/commit/5a900fa9f4fe1c778a7bbf78af3d9b5768f6b4cd))
* **rounding:** Enclose sums near binary64 overflow ([4c04da8](https://github.com/denehoffman/maryada/commit/4c04da8586fe22cb164e868cfe2fcec3c6f86b56))
* **text:** Preserve tight hulls for uncertain literals ([554a4ef](https://github.com/denehoffman/maryada/commit/554a4ef77f649814cdef0fee160b8add89b68c4a))
* **text:** Validate interval endpoint ordering ([1b31ce3](https://github.com/denehoffman/maryada/commit/1b31ce39606e2f5554055f5ef43b84f894fdc0c0))


### Performance improvements

* **bench:** Add Criterion performance benchmarks ([8f09a97](https://github.com/denehoffman/maryada/commit/8f09a972f94bee100d8cb7195a01be18f3e2b735))
* **branch:** Avoid frontier snapshots during gap checks ([44bb9cc](https://github.com/denehoffman/maryada/commit/44bb9ccac9fddc3af341f84b11dd9c8a946315aa))
* **interval:** Evaluate only extremal arithmetic endpoints ([d2b9db0](https://github.com/denehoffman/maryada/commit/d2b9db09aefadbef0841efb05904c0b2f7802a6c))
* **linalg:** Reuse epsilon-inflation setup across solves ([50384b8](https://github.com/denehoffman/maryada/commit/50384b8967b265e5029479163fa24bbe16fd4edc))
* **rounding:** Certify hardware quotients with exact residuals ([21d7bfd](https://github.com/denehoffman/maryada/commit/21d7bfdde79eae5bb4aac7cb49ed4191cad59d2b))

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
