from xdsl.dialects.tpu_memref import EraseLayoutOp
from xdsl.dialects.tpu_shape import BitcastVregOp, ReshapeOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class BitcastVregChainCollapse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BitcastVregOp, rewriter: PatternRewriter) -> None:
        defining_op = op.input.owner
        if not isinstance(defining_op, BitcastVregOp):
            return
        new_op = BitcastVregOp(defining_op.input, op.output.type)
        rewriter.replace_matched_op(new_op)


class ReshapeOfReshape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter) -> None:
        defining_op = op.source.owner
        if not isinstance(defining_op, ReshapeOp):
            return
        new_op = ReshapeOp(defining_op.source, op.result.type)
        rewriter.replace_matched_op(new_op)


class EraseLayoutChainCollapse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EraseLayoutOp, rewriter: PatternRewriter) -> None:
        defining_op = op.operand.owner
        if not isinstance(defining_op, EraseLayoutOp):
            return
        new_op = EraseLayoutOp(defining_op.operand, op.result.type)
        rewriter.replace_matched_op(new_op)
