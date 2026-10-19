"""XDG base-directory helpers (Flatpak-safe via XDG_CONFIG_HOME / XDG_DATA_HOME)."""

import os

APP_DIR_NAME = "vocalinux"


def xdg_config_home() -> str:
    """Return ``$XDG_CONFIG_HOME`` or ``~/.config`` (empty values treated as unset)."""
    return os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")


def xdg_data_home() -> str:
    """Return ``$XDG_DATA_HOME`` or ``~/.local/share`` (empty values treated as unset)."""
    return os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")


def xdg_cache_home() -> str:
    """Return ``$XDG_CACHE_HOME`` or ``~/.cache`` (empty values treated as unset)."""
    return os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")


def xdg_state_home() -> str:
    """Return ``$XDG_STATE_HOME`` or ``~/.local/state`` (empty values treated as unset)."""
    return os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")


def cache_dir() -> str:
    """Return the Vocalinux cache directory."""
    return os.path.join(xdg_cache_home(), APP_DIR_NAME)


def state_dir() -> str:
    """Return the Vocalinux state directory."""
    return os.path.join(xdg_state_home(), APP_DIR_NAME)


def recordings_dir() -> str:
    """Return the directory where dictation recordings are kept for recovery."""
    return os.path.join(cache_dir(), "recordings")


def logs_dir() -> str:
    """Return the directory of the persistent rotating log file."""
    return os.path.join(state_dir(), "logs")


def config_dir() -> str:
    """Return the Vocalinux configuration directory."""
    return os.path.join(xdg_config_home(), APP_DIR_NAME)


def data_dir() -> str:
    """Return the Vocalinux data directory."""
    return os.path.join(xdg_data_home(), APP_DIR_NAME)


def models_dir() -> str:
    """Return the directory where speech-recognition models are stored."""
    return os.path.join(data_dir(), "models")


def is_within_directory(path: str, directory: str, *, allow_root: bool = False) -> bool:
    """Return whether ``path`` resolves inside ``directory``.

    Symlinks are resolved before the check. The directory itself is included
    only when ``allow_root`` is true.
    """
    real_path = os.path.realpath(path)
    real_dir = os.path.realpath(directory)
    try:
        rel = os.path.relpath(real_path, real_dir)
    except ValueError:
        return False
    if os.path.isabs(rel) or rel == ".." or rel.startswith(".." + os.sep):
        return False
    if rel == ".":
        return allow_root
    return True


def collapse_repeated_extension(path: str, extension: str) -> str:
    """Drop a duplicated extension suffix like ``name.txt.txt`` -> ``name.txt``.

    Some save dialogs (notably portal-based ones) append the active filter's
    extension even when the typed name already ends with it.
    """
    doubled = extension + extension
    if path.lower().endswith(doubled):
        return path[: -len(extension)]
    return path
