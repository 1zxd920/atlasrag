# Refund Eligibility

An order can be refunded automatically when payment is confirmed, no shipment
has been dispatched, and the request is within seven calendar days. Other
requests require manual review.

# Failed Refund

After a failed refund, verify the payment transaction identifier and retry only
once with the same idempotency key. If the second attempt fails, stop automated
processing, preserve the error response, and transfer the case to a human.

# Customer Communication

Tell the customer the current status and the next expected update time. Never
promise that a refund completed until the payment provider confirms success.

