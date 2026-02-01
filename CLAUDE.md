# Project: rlm (mahtab)

## Failfast / Minimal Code Philosophy

This codebase follows a strict failfast philosophy. Defensive code creates silent failures that are harder to debug than crashes.

### Rules

1. **NEVER write defensive code**
   - No `try/except` that swallows errors
   - No `getattr(x, 'y', default)` - if the attribute doesn't exist, that's a bug
   - No `if x is not None` guards that hide bugs
   - No `x or default` fallbacks that mask None where None shouldn't be

2. **Fail fast and loud**
   - Let errors propagate with full tracebacks
   - Crashes are better than silent corruption
   - If something unexpected happens, the program should stop

3. **Minimal code**
   - No unnecessary abstractions
   - No "just in case" code
   - No pre-emptive error handling for errors that "might" happen

4. **Trust the caller**
   - Don't validate inputs that shouldn't be invalid
   - If a function requires a string, assume it gets a string
   - Type hints document contracts; runtime checks are redundant

5. **Exceptions to these rules**
   - Network/IO boundaries where failures are expected (and should be handled explicitly)
   - User input validation at system boundaries
   - Resource cleanup in `finally` blocks (but not error swallowing)

### When you see defensive code

If you find code like this, it's a bug to fix:
```python
# BAD - swallows errors
try:
    result = do_thing()
except:
    result = None

# BAD - hides missing attribute bugs
value = getattr(obj, 'attr', default)

# BAD - masks None bugs
data = value or []

# BAD - defensive None check
if result is not None:
    process(result)
```

Replace with direct access that will crash if assumptions are violated.
