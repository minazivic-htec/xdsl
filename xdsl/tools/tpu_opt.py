from xdsl.dialects.tpu import TPU
from xdsl.xdsl_opt_main import xDSLOptMain


class TpuOptMain(xDSLOptMain):
    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.register_dialect(TPU.name, lambda: TPU)


def main():
    TpuOptMain().run()


if __name__ == "__main__":
    main()
