# Swissmetro data

`swissmetro.dat` is the public stated-preference file distributed by the
Transport and Mobility Laboratory at EPFL.

- Source: <https://transp-or.epfl.ch/data/swissmetro.dat>
- Same byte length as the file used in the edX course Choice Models
  (EPFLx ChoiceModels2x).
- SHA-256: `27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`
- 10,728 choice tasks plus a header. Tab-separated.

The file is included so the study can be re-estimated without downloading
it again. It is not covered by the MIT licence on the code. Cite the
original survey:

Bierlaire, M., Axhausen, K., and Abay, G. (2001). The acceptance of modal
innovation: The case of Swissmetro. *1st Swiss Transport Research Conference*.

Variable codes used in the prompts and in the richer logit follow the EPFL
description:

<https://transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf>

Two facts from that description matter for the model:

- If the person holds a General Abonnement, `TRAIN_CO` is the **annual**
  pass price, not the price of the trip. The estimation sample sets the
  trip cost to zero, which is what the Biogeme example does.
- `SM_CO` is a hypothetical fare (about 1.2 times the rail fare) computed
  **without** the pass. The same example sets that trip cost to zero for
  pass holders as well. This repository does the same, and says so in the
  prompt rather than hiding it.

`bash run.sh` checks the checksum and downloads the file only if it is missing
or does not match.
