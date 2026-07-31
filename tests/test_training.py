from ai.trainer import Trainer

trainer = Trainer(

    dataset_path="datasets/crop_disease/train",

    batch_size=8,

    learning_rate=0.001,

    epochs=5

)

trainer.train()