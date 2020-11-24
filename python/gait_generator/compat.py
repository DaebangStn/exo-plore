# Backward compatibility: alias old module paths for checkpoint loading
# Old checkpoints were pickled with 'python.train.*' module paths
# Import this module before loading old checkpoints with dill

import sys

def register_legacy_aliases():
    """Register module aliases for loading old checkpoints."""
    if 'python.train' in sys.modules:
        return  # Already registered

    # Import the actual modules
    import python.gait_generator
    import python.gait_generator.policy
    import python.gait_generator.trainer
    import python.gait_generator.env
    import python.gait_generator.model
    import python.gait_generator.config
    import python.gait_generator.util

    # Create aliases
    sys.modules['python.train'] = python.gait_generator
    sys.modules['python.train.policy'] = python.gait_generator.policy
    sys.modules['python.train.trainer'] = python.gait_generator.trainer
    sys.modules['python.train.env'] = python.gait_generator.env
    sys.modules['python.train.model'] = python.gait_generator.model
    sys.modules['python.train.config'] = python.gait_generator.config
    sys.modules['python.train.util'] = python.gait_generator.util

# Auto-register on import
register_legacy_aliases()
