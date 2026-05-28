from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.arith import ExtFOp, TruncFOp
from xdsl.dialects.builtin import VectorType, bf16, f32
from xdsl.dialects.vector import MultiDimReductionOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


# pravilo
class MultiReductionBitwidthConvert(RewritePattern):
    # MultiDimReductionBitwidthConvert iz jax

    # uzme bf16 vector.multi_reduction pa se u f32 extenduje source i acc bf16 -> f32, reduce u f32, i truncate
    # f32 rez nazad u bf16.

    # ext_src = arith.extf(source) : bf16 vec -> f32 vec
    # ext_acc = arith.extf(acc)    : bf16 vec -> f32 vec
    # red     = vector.multi_reduction(ext_src, ext_acc, kind, dims)

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: MultiDimReductionOp, rewriter: PatternRewriter
    ) -> None:
        src_ty = op.source.type
        # samo bf16 redukcije
        if not (isa(src_ty, VectorType) and src_ty.element_type == bf16):
            return
        res_ty = op.dest.type
        if not isa(res_ty, VectorType):
            return

        # source bf16 -> f32
        src_f32_ty = VectorType(f32, src_ty.get_shape())
        ext_src = ExtFOp(op.source, src_f32_ty)

        # acc bf16 -> f32
        acc_f32_ty = VectorType(f32, res_ty.get_shape())
        ext_acc = ExtFOp(op.acc, acc_f32_ty)

        new_reduction = MultiDimReductionOp(
            ext_src.result,
            ext_acc.result,
            op.kind,
            op.reduction_dims,
            acc_f32_ty,
        )

        # f32 result nazad u bf16
        trunc = TruncFOp(new_reduction.dest, res_ty)

        rewriter.replace_matched_op([ext_src, ext_acc, new_reduction, trunc])


# pass
@dataclass(frozen=True)
class TpuBitwidthConvertPass(ModulePass):
    name = "tpu-bitwidth-convert"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([MultiReductionBitwidthConvert()])
        ).rewrite_module(op)
