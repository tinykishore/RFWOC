global _log


def log(message):
    if _log:
        print("\t"+message)
