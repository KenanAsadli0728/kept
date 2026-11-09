"""Complaint-text templates shared by the seeder (generation) - not used by any model."""
import numpy as np

TYPES = ["late", "damaged", "spoiled", "not_received", "price", "refund_status", "other"]

T = {
 "late": ["my order is late again, I have been waiting since the morning", "still no delivery and it was due hours ago",
          "order hasn't arrived yet, the time slot passed", "delivery is very delayed, where is the driver"],
 "damaged": ["the box arrived crushed and the item is broken", "item arrived damaged, screen cracked",
             "package was torn open and the product is smashed", "glass broken inside the parcel when it came"],
 "spoiled": ["frozen items melted and the food arrived warm and spoiled", "milk and chicken were warm, clearly spoiled",
             "my groceries came warm, ice cream melted everywhere", "cold items arrived not cold, they smell off"],
 "not_received": ["tracking says delivered but nothing arrived at my door", "marked as delivered yet I never received the parcel",
                  "courier scanned it delivered, no package here", "package shows delivered but it is not here"],
 "price": ["the price dropped a few days after I bought it, can you adjust", "I paid more than the current price for this item",
           "saw this cheaper now after buying, want the difference"],
 "refund_status": ["where is my refund, it has been days", "still waiting for the credit you mentioned", "refund status please"],
 "other": ["how do I change my delivery address", "can't log in to my account", "how do I update my payment card", "question about my membership"],
}


def make_text(rng, kind, wh=None, carrier=None, date_str=None, mention_p=0.3):
    base = T[kind][rng.integers(len(T[kind]))]
    CONFUSE = {"spoiled": "damaged", "damaged": "spoiled", "late": "not_received", "not_received": "late"}
    if kind in CONFUSE and rng.random() < 0.50:      # hard cases: the customer mixes in wording from a neighbouring failure type
        base += ", " + T[CONFUSE[kind]][rng.integers(len(T[CONFUSE[kind]]))].split(" and ")[0]
    if kind in ("other", "refund_status") and rng.random() < 0.20:     # hard negatives: failure words in a non-failure ticket
        base = "my last order was late, anyway " + base
    extra = ""
    if kind in ("spoiled", "damaged", "not_received", "late") and rng.random() < mention_p:
        bits = []
        if wh and rng.random() < 0.6: bits.append(f"from warehouse {wh}")
        if carrier and rng.random() < 0.6: bits.append(f"carrier {carrier}")
        if date_str and rng.random() < 0.5: bits.append(f"on {date_str}")
        if bits: extra = " (order " + " ".join(bits) + ")"
    return base + extra
