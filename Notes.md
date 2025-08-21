Each cluster is 16x16 and it has its own embedding (local embedding) just as the whole image itself has an embedding (global embedding)
Example: 224×224 image, 16 → 14×14 = 196 patch embeddings.

#

Possible Things we can do with DINO

1. Just see when we find bugs versus and empty screen. (Automatically delete an image if there are no bugs)
2. See where each "bug" (model does not know) is with the 16x16 cluster.
