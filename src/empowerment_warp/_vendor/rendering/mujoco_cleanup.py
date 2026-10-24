"""MuJoCo renderer cleanup patching for headless/EGL render jobs."""

from __future__ import annotations


_PATCHED = False


def install_safe_mujoco_renderer_cleanup(mujoco_module):
    """Suppress cleanup-only renderer destructor noise on EGL shutdown."""
    global _PATCHED
    if _PATCHED:
        return

    original_close = getattr(mujoco_module.Renderer, 'close', None)
    if original_close is None:
        return

    def _safe_close(self):
        if not hasattr(self, '_gl_context') and not hasattr(self, '_mjr_context'):
            return None
        try:
            return original_close(self)
        except AttributeError as exc:
            if '_mjr_context' not in str(exc) and '_gl_context' not in str(exc):
                raise
        except Exception as exc:
            cleanup_typeerror = (
                exc.__class__.__name__ == 'TypeError'
                and 'NoneType' in str(exc)
                and 'not callable' in str(exc)
            )
            if exc.__class__.__name__ not in {'EGLError', 'GLError'} and not cleanup_typeerror:
                raise
        finally:
            for attr_name in ('_gl_context', '_mjr_context'):
                if hasattr(self, attr_name):
                    try:
                        setattr(self, attr_name, None)
                    except Exception:
                        pass
        return None

    mujoco_module.Renderer.close = _safe_close
    mujoco_module.Renderer.__del__ = _safe_close
    try:
        from mujoco import egl as mujoco_egl
    except Exception:
        mujoco_egl = None
    gl_context_cls = getattr(mujoco_egl, 'GLContext', None) if mujoco_egl is not None else None
    original_gl_free = getattr(gl_context_cls, 'free', None) if gl_context_cls is not None else None
    if original_gl_free is not None:
        def _safe_gl_free(self):
            if not hasattr(self, '_context'):
                return None
            try:
                return original_gl_free(self)
            except Exception as exc:
                cleanup_typeerror = (
                    exc.__class__.__name__ == 'TypeError'
                    and 'NoneType' in str(exc)
                    and 'not callable' in str(exc)
                )
                if exc.__class__.__name__ not in {'EGLError', 'GLError'} and not cleanup_typeerror:
                    raise
            finally:
                if hasattr(self, '_context'):
                    try:
                        self._context = None
                    except Exception:
                        pass
            return None

        gl_context_cls.free = _safe_gl_free
        gl_context_cls.__del__ = _safe_gl_free
    _PATCHED = True
