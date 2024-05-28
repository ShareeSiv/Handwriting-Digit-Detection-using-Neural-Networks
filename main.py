from train_eval import train, test

def main():
    for epoch in range(1, 11):  # Train for 10 epochs
        train(epoch)
        test()

if __name__ == "__main__":
    main()