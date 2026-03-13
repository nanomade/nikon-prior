"""Return a motor manager: real PriorMotorManager if hardware is found,
otherwise MockMotorManager for offline/development use."""

import logging

logger = logging.getLogger(__name__)


def create_motor_manager():
    try:
        from motors.prior_motor_manager import PriorMotorManager
        mm = PriorMotorManager()
        mm.connect()
        logger.info("ProScan III connected on %s", mm.port)
        return mm
    except Exception as exc:
        logger.warning("ProScan III not available (%s) — using mock motors", exc)
        from motors.mock_manager import MockMotorManager
        return MockMotorManager()
