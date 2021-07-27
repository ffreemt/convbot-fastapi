"""Run __main__."

org deepl-fastapi
Run uvicorn with deepl_fastapi.deepl_server:app.

uvicorn deepl_fastapi.deepl_server:app --reload
"""
import uvicorn

# import portalocker
# from signal import signal, SIGINT, SIG_DFL

from logzero import logger


def run_uvicorn(host="0.0.0.0", port=8000, debug=False, reload=False):
    uvicorn.run(
        app="convbot_fastapi.convbot_fastapi:app",
        host=host,
        port=port,
        debug=debug,
        reload=reload,
        # workers=2,
        # loop="asyncio",  # default "auto"
        # loop="uvloop",  # posix (linux and mac) only
    )


def main():
    # signal(SIGINT, SIG_DFL)
    # print("ctrl-C to interrupt")

    _ = """
    file_ = Path(__file__).parent / "deepl_server.py"
    lockfile = Path(f"{file_}.portalocker.lock")
    if not Path(lockfile).exists():
        Path(lockfile).touch()
    try:
        file = open(lockfile, "r+")
        # portalocker.lock(file, portalocker.constants.LOCK_EX)
        portalocker.lock(file, portalocker.LOCK_EX | portalocker.LOCK_NB)
        ...
    except Exception as exc:
        logger.debug(exc)
        logger.error("Another copy is running, exiting...")
        raise SystemExit(1) from exc
        # raise
    finally:
        # LOOP.close()
        ...
    """

    try:
        run_uvicorn()
    except Exception as exc:
        logger.error(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
