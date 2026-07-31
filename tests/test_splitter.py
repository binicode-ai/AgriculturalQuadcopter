from ai.splitter import DatasetSplitter

splitter = DatasetSplitter(

    source_dir="datasets",

    destination_dir="datasets/crop_disease"

)

splitter.split()