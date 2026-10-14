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
