//! The system-call dispatcher: the only code that takes kernel locks. It acquires the tables a call
//! declares, in one fixed order, and records the event before releasing them.
