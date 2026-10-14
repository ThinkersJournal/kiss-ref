# Vendored KISS conformance corpus — provenance

Source: ThinkersJournal/KISS `conformance/corpus/` at `origin/main` `9ea8c05` (2026-10-14; KISS 0.2.14; the two integer/bool files below are from `034a4b1`, KISS 0.2.20; `sk5` / `contract_version` 3). Copied verbatim; do not edit. Re-vendor by copying again and updating this table.

| file | KISS blob (`git rev-parse 9ea8c05:conformance/corpus/<file>`) |
|---|---|
| `dtype_manifest.json` | `24ff38f068aa18be723c07d307c86ae572c1333f` (changed from `af851db` at `904a4b4`: only `structure_key_schema_version` 4→5 and `token_prefix` `sk4`→`sk5`; the 24 dtype rows are byte-identical) |
| `op_manifest.json` | `f656b625ab8e5371c7a4a73df92863d193c31b74` (unchanged from `904a4b4`) |
| `ops-arith.json` | `aee2de79a709e722f3c439047bf90c638aec9b24` |
| `ops-minmax-ordinary.json` | `7bd6cc394bd247f9ca34edfc957d70a37432af97` |
| `ops-minmax-signed-zero.json` | `d06bdb85c8e626750318b61585b4af3ea5c8d4b4` |
| `ops-narrow-move-nan.json` | `c108226af556aa1c6a45e8e43b85ab29b96dec2d` |
| `ops-narrow-select-nan.json` | `2efbfd259c43936d8cf6dc24b1375f6c9b5ae81b` |
| `ops-transcendental-nan.json` | `d4f8b7486ca2b082252f4d305dff8edff2db5672` (13 vectors: +f64 `log` of an sNaN, tcId 13, KISS #550; was `7e46250` / 12; also gains a `harness_rule` header key) |
| `ops-minmax-nan.json` | `576d69ab693f6acb835c1c91307dba26bb0390d3` (new at KISS #540: 44 f32/f64 NaN cells for max_prop/min_prop/fmax_ieee/fmin_ieee) |
| `ops-rounding.json` | `f8f84610bc59bffe4e77c1ea2546482668fe5d77` (new at KISS #539: 72 f32 floor/ceil/trunc/round_even cells) |
| `ops-int-wrap.json` | `00d16c351833e404bf8ffd80a4b62d778e0f0b84` (new at KISS 0.2.19, vendored from `origin/main` `034a4b1`: 440 exact-byte i8..u64 add/sub/mul wrap cells, KISS-OPS-6.2-0002) |
| `ops-compare-bool.json` | `761bee795394b60936781a0a569bac0c50f47f2c` (new at KISS 0.2.20, vendored from `origin/main` `034a4b1`: 181 exact-byte cells, `cmp_*` over f32/i32/u32 and `logical_*` over `bool`, expected dtype `bool`, KISS-OPS-6.2-0005/-0006) |
| `ops-atan2-quadrants.json` | `0c851c27b1702d46b6147a0b0c5b86893a9da95b` (new at KISS 0.2.22 / #558, vendored from `origin/main` `d83d3c0`: 28 f32 class-ULP `atan2` cells, KISS-OPS-6.9-0001; expected is f64 libm `atan2` rounded to f32, not correctly rounded) |
| `ops-sqrt-rounding.json` | `14669a5c0551bc1ee73062f35ecc9cda8ec4b410` (new at KISS 0.2.23 / #559, vendored from `origin/main` `6aeb9b6`: 24 f32 class-ULP `sqrt` cells with `ulp_bound` 0, KISS-OPS-6.8-0003 first arm only; expected is `f32::sqrt`, IEEE correctly rounded) |
| `ops-nextafter.json` | `f0f538e37974a488d5dfaa23eff680d6ceac2d82` (new at KISS 0.2.26 / #562, vendored from `origin/main` `e317315`: 40 exact-byte f32/f64 `nextafter` cells, KISS-OPS-6.9-0003 value half only; expected from KISS's sign-magnitude stepping, cross-checked there by an independent monotone-key model) |
