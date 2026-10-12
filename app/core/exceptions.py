"""Domain exceptions. Each carries the HTTP status + machine-readable code the API returns."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class EmptyDocumentError(AppError):
    status_code = 422
    code = "empty_document"


class CorruptDocumentError(AppError):
    status_code = 422
    code = "corrupt_document"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


class LLMNotConfiguredError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMUnavailableError(AppError):
    status_code = 502
    code = "llm_unavailable"


class LLMResponseError(AppError):
    status_code = 502
    code = "llm_invalid_response"
