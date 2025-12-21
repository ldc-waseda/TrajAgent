"""
OpenAI Client Wrapper - Compatibility shim for openai_batch_wrapper package.

This module re-exports from the openai_batch_wrapper package for backward compatibility.
For new code, import directly from openai_batch_wrapper.

Example:
    # Old way (still works)
    from visual_language.utils.openai_wrapper import create_wrapped_client
    
    # New way (recommended)
    from openai_batch_wrapper import create_wrapped_client
"""

try:
    # Try to import from the standalone package
    from openai_batch_wrapper import (
        WrappedAsyncOpenAI,
        WrappedChat,
        WrappedChatCompletions,
        MockChatCompletion,
        create_wrapped_client,
        get_queue_stats,
        CacheManager,
        BatchQueueManager,
        BatchManager,
        ProviderAdapter,
        OpenAIAdapter,
        get_adapter,
        convert_batch_request,
        convert_batch_response,
    )
    
    __all__ = [
        "WrappedAsyncOpenAI",
        "WrappedChat", 
        "WrappedChatCompletions",
        "MockChatCompletion",
        "create_wrapped_client",
        "get_queue_stats",
        "CacheManager",
        "BatchQueueManager",
        "BatchManager",
        "ProviderAdapter",
        "OpenAIAdapter",
        "get_adapter",
        "convert_batch_request",
        "convert_batch_response",
    ]
    
except ImportError:
    # Fallback to local implementation if package not installed
    # This allows the code to work during development before the package is installed
    import warnings
    warnings.warn(
        "openai_batch_wrapper package not found, using local implementation. "
        "Install with: pip install -e /path/to/openai_batch_wrapper",
        ImportWarning
    )
    
    # Import local implementation
    from ._openai_wrapper_impl import (
        WrappedAsyncOpenAI,
        WrappedChat,
        WrappedChatCompletions,
        MockChatCompletion,
        create_wrapped_client,
        get_queue_stats,
    )
    from .cache_manager import CacheManager
    from .batch_manager import BatchManager
    from .provider_adapters import (
        ProviderAdapter,
        OpenAIAdapter,
        get_adapter,
        convert_batch_request,
        convert_batch_response,
    )
    
    # BatchQueueManager needs to be imported from wrapper impl
    from ._openai_wrapper_impl import BatchQueueManager
    
    __all__ = [
        "WrappedAsyncOpenAI",
        "WrappedChat",
        "WrappedChatCompletions", 
        "MockChatCompletion",
        "create_wrapped_client",
        "get_queue_stats",
        "CacheManager",
        "BatchQueueManager",
        "BatchManager",
        "ProviderAdapter",
        "OpenAIAdapter",
        "get_adapter",
        "convert_batch_request",
        "convert_batch_response",
    ]
