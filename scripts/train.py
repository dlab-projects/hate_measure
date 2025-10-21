from hate_measure.train import HateSpeechTrainer, create_default_config

def main():
    # Load default config
    config = create_default_config()

    # Initialize trainer
    trainer = HateSpeechTrainer(config)

    # Prepare data and train
    trainer.prepare_data()
    trainer.train()

    # Save final model
    trainer.save_model("final_hate_speech_model.pt")

if __name__ == "__main__":
    main()
