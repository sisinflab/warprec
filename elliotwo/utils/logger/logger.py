from loguru import logger as loguru_logger
import sys

# Define custom logging levels
MSG = 10
POSITIVE_MSG = 11
NEGATIVE_MSG = 12
STATS = 13
ATTENTION = 14

# Register custom levels in loguru with color and icon
loguru_logger.level("CONSOLE_MSG", no=MSG, color="<white>", icon="📝")
loguru_logger.level("POSITIVE_MSG", no=POSITIVE_MSG, color="<green>", icon="✅")
loguru_logger.level("NEGATIVE_MSG", no=NEGATIVE_MSG, color="<red>", icon="❌")
loguru_logger.level("STATS", no=STATS, color="<cyan>", icon="📊")
loguru_logger.level("ATTENTION", no=ATTENTION, color="<yellow>", icon="📢")


class CustomLogger:
    """A wrapper for loguru that replicates a custom logging interface with icons and colors.

    This class configures loguru to output logs with a specified format and provides helper methods
    to log messages with custom levels. All log records are bound with a custom 'name' to ensure the
    formatter's expectations are met.

    When initialized removes default loguru sinks, binds the logger with a given name (ensuring that every
    log record contains the extra field 'name'), and adds a new sink that logs to sys.stdout using a
    custom format including time, logger name, level icon, level name (for color), and message. Colors
    are enabled using the 'colorize' parameter.

    Args:
        name (str): The name of the logger.
        level (str, optional): The minimum log level to capture. Defaults to "DEBUG".
    """

    def __init__(self, name: str, level: str = "DEBUG"):
        self.name = name
        # Remove default sinks to avoid duplicate logs
        loguru_logger.remove()
        # Bind the logger with the custom name so that the extra field is always set.
        self._logger = loguru_logger.bind(name=self.name)
        # Add a new sink with a custom format that includes the level (which carries color info).
        loguru_logger.add(
            sys.stdout,
            format="{time:DD-MM-YYYY HH:mm:ss} - {extra[name]} - {level.icon} <level>{message}</level>",
            level=level,
            colorize=True,
        )

    def msg(self, message: str, *args, **kwargs):
        """Logs a message with the CONSOLE_MSG level.

        Args:
            message (str): The log message.
            *args: Additional positional arguments passed to the logger.
            **kwargs: Additional keyword arguments passed to the logger.
        """
        self._logger.log("CONSOLE_MSG", message, *args, **kwargs)

    def positive(self, message: str, *args, **kwargs):
        """Logs a message with the POSITIVE_MSG level.

        Args:
            message (str): The log message.
            *args: Additional positional arguments passed to the logger.
            **kwargs: Additional keyword arguments passed to the logger.
        """
        self._logger.log("POSITIVE_MSG", message, *args, **kwargs)

    def negative(self, message: str, *args, **kwargs):
        """Logs a message with the NEGATIVE_MSG level.

        Args:
            message (str): The log message.
            *args: Additional positional arguments passed to the logger.
            **kwargs: Additional keyword arguments passed to the logger.
        """
        self._logger.log("NEGATIVE_MSG", message, *args, **kwargs)

    def stats(self, message: str, *args, **kwargs):
        """Logs a message with the STATS level.

        Args:
            message (str): The log message.
            *args: Additional positional arguments passed to the logger.
            **kwargs: Additional keyword arguments passed to the logger.
        """
        self._logger.log("STATS", message, *args, **kwargs)

    def attention(self, message: str, *args, **kwargs):
        """Logs a message with the ATTENTION level.

        Args:
            message (str): The log message.
            *args: Additional positional arguments passed to the logger.
            **kwargs: Additional keyword arguments passed to the logger.
        """
        self._logger.log("ATTENTION", message, *args, **kwargs)

    def separator(self, length: int = 80):
        """Logs a separator line using the CONSOLE_MSG level.

        Args:
            length (int, optional): The number of characters in the separator. Defaults to 80.
        """
        self.attention("-" * length)

    def stat_msg(self, message: str, header: str):
        """Logs a statistical message with a header and separator.

        This method logs a header centered with dashes, then logs the message at the STATS level,
        and finally logs a separator line.

        Args:
            message (str): The statistical message.
            header (str): The header for the statistical message.
        """
        self.msg(header.center(len(message), "-"))
        self.stats(message)
        self.msg("-" * len(message))


# Create a singleton instance of CustomLogger
logger = CustomLogger("Elliotwo")
