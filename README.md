# **Drug discovery project about ligands that bind to Beta-Lactamases**

## **About the project**

This project aims to evaluate the activity of molecules that have been experimentally tested to bind or not bind to [Beta-Lactamases](https://en.wikipedia.org/wiki/Beta-lactamase). Some of these proteins allow multi-drug resistant bacteria or superbugs to inactivate a wide range of penicillin-like antibiotics, which is known as antimicrobial resistance (AMR). According to the World Health Organization, AMR is one of the [top ten global public health threats facing humanity in this century](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance), so it is important to search for potential compounds that combat these superbugs and prevent AMR. You can find detailed information about AMR and Beta-Lactamase in this [blog](https://pdb101.rcsb.org/motm/187).

## **Dataset**

The [dataset](https://www.kaggle.com/thedataprof/betalactamase) consists of 136 csv files with information of interactions between small molecules and Beta-Lactamases. These features are listed below:

* `molecule_chembl_id:` unique CHEMBL identifier of the molecule.
* `canonical_smiles:` one-dimensional representation of chemical structure from the molecule.
* `standard_value:` bioactivity value, which was experimentally measured. Standard value can be defined as the concentration of a drug, and if this value is lower the drug has better bioactivity because it requires low concentration to have an effect. This can be the target variable for machine learning models.
* `standard_relation:` this value tells if the standard value was reported as a finite number or if this value is greater than or lower than a particular number.
* `standard_units:` units of the bioactivity value.
* `standard_type:` type of experimental assay used to measure the bioactivity value.
* `pchembl_value:` bioactivity value that combines measurements from IC50 and Ki assays, and applies a negative logarithmic transformation. This can be the target variable for machine learning models.
* `target_pref_name:` name of protein tested to interact with the molecule.
* `bao_label:` bioactivity experiment standard label.

## **Data preparation and feature matrix**

The feature matrix to train machine learning models is obtained by calculating molecular descriptors from the `canonical_smiles` of molecules. These molecular descriptors are also known as molecular fingerprints, and they are property profiles of molecules, represented as vectors with each vector element representing the existence or the frequency of a structural feature. The extraction of molecular fingerprints from SMILES should be performed with [PaDEL](http://www.yapcwsoft.com/dd/padeldescriptor/) software, following instructions from [this video](https://youtu.be/rEmDyZHz5U8).

In addition, it is important to notice that this dataset contains interaction data of molecules with various Beta-lactamases. Therefore, we can create independent machine learning models for each of these proteins, or a single unified model for all of them, which is known as the proteochemometric model.

## **How to contribute with the Open Bioinformatics Research Project?**

More information about this project and how to contribute to it in this [video](https://youtu.be/_GtEgiWWyK4).
