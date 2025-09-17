from utils.calcoli import somma, fattoriale


def main():
    result = somma(3, 5)
    print(f"La somma è: {result}")
    fact = fattoriale(5)
    print(f"Il fattoriale è: {fact}")


if __name__ == "__main__":
    main()
