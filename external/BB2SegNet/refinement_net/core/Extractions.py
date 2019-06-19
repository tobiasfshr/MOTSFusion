
EXTRACTIONS = "extractions"

SEGMENTATION_POSTERIORS = "segmentation_posteriors"
SEGMENTATION_POSTERIORS_ORIGINAL_SIZE = "segmentation_posteriors_original_size"
SEGMENTATION_MASK_ORIGINAL_SIZE = "segmentation_mask_original_size"
SEGMENTATION_MASK_INPUT_SIZE = "segmentation_mask_input_size"


def accumulate_extractions(extractions_accumulator, *new_extractions):
  if len(new_extractions) == 0:
    return

  if len(extractions_accumulator) == 0:
    extractions_accumulator.update(new_extractions[0])
    new_extractions = new_extractions[1:]

  # each extraction will actually be a list, so we can just sum up the lists (extend the accumulator list with the next)
  for k, v in extractions_accumulator.items():
    for ext in new_extractions:
      extractions_accumulator[k] += ext[k]

  return extractions_accumulator
