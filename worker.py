from __future__ import annotations

import logging

from whisper_transcriber.worker import main


if __name__ == "__main__":
    console = logging.StreamHandler()
    # Job logs record DEBUG output; keep the console at INFO.
    console.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[console],
    )
    raise SystemExit(main())

