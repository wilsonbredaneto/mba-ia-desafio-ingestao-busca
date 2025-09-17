from search import search_prompt

def main():
   
    while True:
        print("\nFaça sua pergunta:")
        pergunta = input("\nPERGUNTA: ").strip()
        resposta = search_prompt(pergunta)
        print(f"RESPOSTA: {resposta}")           

if __name__ == "__main__":
    main()