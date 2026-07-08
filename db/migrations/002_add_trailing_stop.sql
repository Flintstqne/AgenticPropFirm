-- Adds trailing_stop to trades: the price distance a stop trails behind the
-- best price reached since entry. NULL means no trailing stop on that trade,
-- stop_loss behaves as a fixed level as before.
ALTER TABLE trades ADD COLUMN trailing_stop REAL;
