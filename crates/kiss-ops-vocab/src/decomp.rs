// SPDX-License-Identifier: MIT OR Apache-2.0
//! §6.13 reference decompositions of non-primitive ops.
//!
//! Each non-primitive op resolves to the primitive floor via the reference
//! decomposition in KISS-Ops §6.13 (KISS-OPS-6.13-0001). The decompositions are
//! stored here as the spec's own expression strings (verbatim from the §6.13
//! table) so they are auditable against — and regenerable from — the spec, and
//! parsed into an [`Expr`] tree by [`parse`] following the §6.13-0006 grammar.
//!
//! Operand naming follows §6.13-0004: the spec's `x` and `a` are `input(0)`, `b`
//! is `input(1)`.
//!
//! **Seed scope.** This module binds the **elementwise** subset of §6.13 (every
//! decomposition that resolves purely through scalar floor atoms). The
//! reduction / scan / normalization / contraction / window / gather-scatter
//! non-primitives (which decompose through the structural atoms `reduce` /
//! `prefix_scan` / `gather` / `sort_network` and the §6.13-0006 let-binding body
//! form) are enumerated in the vocabulary but their decomposition is `None` here
//! — a documented PENDING region for the next wave. The coverage gate reports
//! them.

extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::Op;

/// A pinned symbolic / literal constant used in a §6.13 decomposition
/// (`const(...)`). Each is a `const(bits)` leaf pinned per §6.12-0003.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstSym {
    /// `const(0)`
    Zero,
    /// `const(1)`
    One,
    /// `const(-1)`
    NegOne,
    /// `const(0.5)`
    Half,
    /// `const(2)`
    Two,
    /// `const(3)`
    Three,
    /// `const(ln2)`
    Ln2,
    /// `const(ln10)`
    Ln10,
    /// `const(pi/2)`
    PiOver2,
    /// `const(sqrt2)`
    Sqrt2,
    /// `const(sqrt(2/pi))`
    SqrtTwoOverPi,
    /// `const(0.044715)` — the `gelu_tanh` polynomial coefficient.
    C0_044715,
}

impl ConstSym {
    /// The `f64` value of this constant (the reference evaluates in the compute
    /// dtype; `f64` is the widest reference).
    pub const fn value(self) -> f64 {
        use core::f64::consts;
        match self {
            ConstSym::Zero => 0.0,
            ConstSym::One => 1.0,
            ConstSym::NegOne => -1.0,
            ConstSym::Half => 0.5,
            ConstSym::Two => 2.0,
            ConstSym::Three => 3.0,
            ConstSym::Ln2 => consts::LN_2,
            ConstSym::Ln10 => consts::LN_10,
            ConstSym::PiOver2 => consts::FRAC_PI_2,
            ConstSym::Sqrt2 => consts::SQRT_2,
            // sqrt(2/pi) pinned as a const(bits) leaf (§6.12-0003) so the vocab
            // crate stays dependency-free (no libm for a runtime sqrt).
            ConstSym::SqrtTwoOverPi => 0.797_884_560_802_865_4,
            ConstSym::C0_044715 => 0.044715,
        }
    }

    fn from_inner(s: &str) -> Option<ConstSym> {
        match s.trim() {
            "0" | "0.0" => Some(ConstSym::Zero),
            "1" | "1.0" => Some(ConstSym::One),
            "-1" | "-1.0" => Some(ConstSym::NegOne),
            "0.5" => Some(ConstSym::Half),
            "2" | "2.0" => Some(ConstSym::Two),
            "3" | "3.0" => Some(ConstSym::Three),
            "ln2" => Some(ConstSym::Ln2),
            "ln10" => Some(ConstSym::Ln10),
            "pi/2" => Some(ConstSym::PiOver2),
            "sqrt2" => Some(ConstSym::Sqrt2),
            "sqrt(2/pi)" => Some(ConstSym::SqrtTwoOverPi),
            "0.044715" => Some(ConstSym::C0_044715),
            _ => None,
        }
    }
}

/// A parsed §6.13 reference-decomposition expression tree (§6.13-0006 grammar,
/// single-expression form).
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// An operand leaf: `input(0)` (spelled `x` / `a` in the spec) or
    /// `input(1)` (`b`).
    Input(u8),
    /// A `const(...)` leaf.
    Const(ConstSym),
    /// A KISS-Ops op applied to argument sub-expressions.
    Apply(Op, Vec<Expr>),
}

/// A parse failure with a byte position into the source.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub msg: String,
    pub pos: usize,
}

/// Parse a §6.13 decomposition body in the single-expression form of the
/// §6.13-0006 grammar. (The let-binding form is not needed by the elementwise
/// subset and is a documented follow-up.)
pub fn parse(src: &str) -> Result<Expr, ParseError> {
    let mut p = Parser {
        s: src.as_bytes(),
        i: 0,
    };
    p.skip_ws();
    let e = p.expr()?;
    p.skip_ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing input after expression"));
    }
    Ok(e)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, m: &str) -> ParseError {
        ParseError {
            msg: m.to_string(),
            pos: self.i,
        }
    }

    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    /// An identifier token: `[A-Za-z0-9_]+` (also matches the leading `-` of a
    /// negative numeric is NOT handled here — negatives only appear inside
    /// `const(...)` which is captured whole).
    fn ident(&mut self) -> &'a str {
        let start = self.i;
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.i += 1;
            } else {
                break;
            }
        }
        core::str::from_utf8(&self.s[start..self.i]).unwrap_or("")
    }

    /// Capture the balanced-paren content immediately following (the `(` must be
    /// the current char). Returns the inner text without the outer parens.
    fn balanced(&mut self) -> Result<&'a str, ParseError> {
        if self.peek() != Some(b'(') {
            return Err(self.err("expected '('"));
        }
        self.i += 1; // consume '('
        let start = self.i;
        let mut depth = 1usize;
        while self.i < self.s.len() {
            match self.s[self.i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        let inner = core::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
                        self.i += 1; // consume ')'
                        return Ok(inner);
                    }
                }
                _ => {}
            }
            self.i += 1;
        }
        Err(self.err("unbalanced parentheses"))
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        self.skip_ws();
        // A bare operand identifier (x/a/b) or a head identifier followed by '('.
        let head = self.ident();
        if head.is_empty() {
            return Err(self.err("expected an op token, const, input, or operand name"));
        }
        self.skip_ws();
        if self.peek() != Some(b'(') {
            // Bare operand name.
            return match head {
                "x" | "a" => Ok(Expr::Input(0)),
                "b" => Ok(Expr::Input(1)),
                other => Err(ParseError {
                    msg: alloc::format!("unknown bare operand '{other}'"),
                    pos: self.i,
                }),
            };
        }
        match head {
            "const" => {
                let inner = self.balanced()?;
                match ConstSym::from_inner(inner) {
                    Some(c) => Ok(Expr::Const(c)),
                    None => Err(ParseError {
                        msg: alloc::format!("unknown const '{}'", inner.trim()),
                        pos: self.i,
                    }),
                }
            }
            "input" => {
                let inner = self.balanced()?;
                match inner.trim().parse::<u8>() {
                    Ok(n) => Ok(Expr::Input(n)),
                    Err(_) => Err(ParseError {
                        msg: alloc::format!("bad input index '{}'", inner.trim()),
                        pos: self.i,
                    }),
                }
            }
            op_tok => {
                let op = Op::from_token(op_tok).ok_or_else(|| ParseError {
                    msg: alloc::format!("unknown op token '{op_tok}'"),
                    pos: self.i,
                })?;
                let args = self.arg_list()?;
                Ok(Expr::Apply(op, args))
            }
        }
    }

    /// Parse `( expr (, expr)* )` (or `()`), the current char being `(`.
    fn arg_list(&mut self) -> Result<Vec<Expr>, ParseError> {
        if self.peek() != Some(b'(') {
            return Err(self.err("expected '('"));
        }
        self.i += 1; // consume '('
        let mut args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b')') {
            self.i += 1;
            return Ok(args);
        }
        loop {
            let e = self.expr()?;
            args.push(e);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    self.skip_ws();
                }
                Some(b')') => {
                    self.i += 1;
                    return Ok(args);
                }
                _ => return Err(self.err("expected ',' or ')'")),
            }
        }
    }
}

impl Op {
    /// The verbatim §6.13 reference-decomposition source string for this op, or
    /// `None` for a primitive-floor op (KISS-OPS-6.3-0003: floor ops carry no
    /// decomposition) or a non-primitive whose decomposition is not yet bound in
    /// this seed (the structural-atom / let-body PENDING region).
    ///
    /// Strings are copied verbatim from the KISS-Ops §6.13 table; `x`/`a` are
    /// `input(0)`, `b` is `input(1)` (§6.13-0004).
    pub fn reference_decomposition_src(self) -> Option<&'static str> {
        Some(match self {
            Op::Sqr => "mul(x, x)",
            Op::Recip => "div(const(1), x)",
            Op::Rsqrt => "div(const(1), sqrt(x))",
            Op::Frac => "sub(x, trunc(x))",
            Op::Sign => {
                "select(cmp_gt(x,const(0)), const(1), select(cmp_lt(x,const(0)), const(-1), const(0)))"
            }
            Op::Step => "select(cmp_gt(x, const(0)), const(1), const(0))",
            Op::MaxProp => {
                "select(cmp_ne(a,a), a, select(cmp_ne(b,b), b, select(cmp_ge(a,b), a, b)))"
            }
            Op::MinProp => {
                "select(cmp_ne(a,a), a, select(cmp_ne(b,b), b, select(cmp_le(a,b), a, b)))"
            }
            Op::FmaxIeee => {
                "select(cmp_ne(a,a), b, select(cmp_ne(b,b), a, select(cmp_ge(a,b), a, b)))"
            }
            Op::FminIeee => {
                "select(cmp_ne(a,a), b, select(cmp_ne(b,b), a, select(cmp_le(a,b), a, b)))"
            }
            Op::Exp2 => "exp(mul(x, const(ln2)))",
            Op::Expm1 => "sub(exp(x), const(1))",
            Op::Log2 => "div(log(x), const(ln2))",
            Op::Log10 => "div(log(x), const(ln10))",
            Op::Log1p => "log(add(const(1), x))",
            Op::Tan => "div(sin(x), cos(x))",
            Op::Tanh => "div(sub(exp(x), exp(neg(x))), add(exp(x), exp(neg(x))))",
            Op::Sinh => "div(sub(exp(x), exp(neg(x))), const(2))",
            Op::Cosh => "div(add(exp(x), exp(neg(x))), const(2))",
            Op::Asinh => "log(add(x, sqrt(add(sqr(x), const(1)))))",
            Op::Acosh => "log(add(x, sqrt(sub(sqr(x), const(1)))))",
            Op::Atanh => "mul(const(0.5), log(div(add(const(1),x), sub(const(1),x))))",
            Op::Asin => "atan(div(x, sqrt(sub(const(1), sqr(x)))))",
            Op::Acos => "sub(const(pi/2), asin(x))",
            Op::Cbrt => "mul(sign(x), exp(div(log(abs(x)), const(3))))",
            Op::Erfc => "sub(const(1), erf(x))",
            Op::Sigmoid => "recip(add(const(1), exp(neg(x))))",
            Op::Relu => "select(cmp_lt(x, const(0)), const(0), x)",
            Op::Silu => "mul(x, sigmoid(x))",
            Op::Softplus => "log(add(const(1), exp(x)))",
            Op::Mish => "mul(x, tanh(softplus(x)))",
            Op::Gelu => "mul(mul(const(0.5), x), add(const(1), erf(div(x, const(sqrt2)))))",
            Op::GeluTanh => {
                "mul(mul(const(0.5), x), add(const(1), tanh(mul(const(sqrt(2/pi)), add(x, mul(const(0.044715), mul(x, sqr(x))))))))"
            }
            Op::Pow => "exp(mul(b, log(a)))",
            Op::Hypot => "sqrt(add(sqr(a), sqr(b)))",
            Op::RemFloor => "sub(a, mul(floor(div(a,b)), b))",
            Op::RemTrunc => "sub(a, mul(trunc(div(a,b)), b))",
            Op::Ldexp => "mul(a, exp2(b))",
            Op::LogicalAnd => "mul(cmp_ne(a, const(0)), cmp_ne(b, const(0)))",
            Op::LogicalOr => {
                "min_prop(const(1), add(cmp_ne(a,const(0)), cmp_ne(b,const(0))))"
            }
            Op::LogicalNot => "cmp_eq(x, const(0))",
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomp_floor_ops_have_no_decomposition() {
        // KISS-OPS-6.3-0003.
        for o in Op::ALL.iter().filter(|o| o.is_primitive_floor()) {
            assert_eq!(
                o.reference_decomposition_src(),
                None,
                "floor op {o:?} must carry no decomposition"
            );
        }
    }

    #[test]
    fn decomp_all_bound_sources_parse() {
        // Born-red guard: a typo in any verbatim source is caught here.
        for o in Op::ALL {
            if let Some(src) = o.reference_decomposition_src() {
                parse(src).unwrap_or_else(|e| panic!("{o:?} src {src:?} failed to parse: {e:?}"));
            }
        }
    }

    #[test]
    fn decomp_gelu_parses_to_expected_tree() {
        let e = parse(Op::Gelu.reference_decomposition_src().unwrap()).unwrap();
        // mul(mul(const(0.5), x), add(const(1), erf(div(x, const(sqrt2)))))
        match e {
            Expr::Apply(Op::Mul, ref args) => {
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expr::Apply(Op::Mul, _)));
                assert!(matches!(args[1], Expr::Apply(Op::Add, _)));
            }
            other => panic!("gelu root should be mul, got {other:?}"),
        }
    }

    #[test]
    fn decomp_no_op_references_itself_directly() {
        // KISS-OPS-6.14-0005 (direct case): an op MUST NOT appear in its own
        // reference decomposition.
        fn mentions(e: &Expr, target: Op) -> bool {
            match e {
                Expr::Apply(op, args) => *op == target || args.iter().any(|a| mentions(a, target)),
                _ => false,
            }
        }
        for o in Op::ALL {
            if let Some(src) = o.reference_decomposition_src() {
                let e = parse(src).unwrap();
                assert!(
                    !mentions(&e, *o),
                    "{o:?} references itself in its decomposition"
                );
            }
        }
    }

    #[test]
    fn decomp_bound_ops_resolve_to_floor() {
        // KISS-OPS-6.14: recursive resolution of a bound non-primitive terminates
        // at the primitive floor. We expand references, bounded by a depth budget;
        // any op with a bound decomposition is expanded, floor atoms are the base.
        fn resolves(op: Op, budget: u32) -> bool {
            if budget == 0 {
                return false;
            }
            if op.is_primitive_floor() {
                return true;
            }
            let Some(src) = op.reference_decomposition_src() else {
                // A non-primitive with no bound decomposition (PENDING region) is
                // out of scope for this closure check.
                return true;
            };
            let e = parse(src).unwrap();
            fn walk(e: &Expr, budget: u32) -> bool {
                match e {
                    Expr::Input(_) | Expr::Const(_) => true,
                    Expr::Apply(op, args) => {
                        resolves(*op, budget - 1) && args.iter().all(|a| walk(a, budget))
                    }
                }
            }
            walk(&e, budget)
        }
        for o in Op::ALL {
            if o.reference_decomposition_src().is_some() {
                assert!(
                    resolves(*o, 32),
                    "{o:?} did not resolve to floor within budget"
                );
            }
        }
    }
}
