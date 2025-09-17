def somma(a, b):
    """
    Calcola la somma tra due numeri interi

    Parameters
    ----------
    a : int
        Numero intero non negativo di cui calcolare il fattoriale.
    b : int
        Numero intero non negativo di cui calcolare il fattoriale.
    Returns
    -------
    int
        La somma di due numeri
    """
    return a + b

def fattoriale(n) -> int:
    """
    Calcola il fattoriale di un numero n (n!)
    
    Parameters
    ----------
    n : int
        Numero intero non negativo di cui calcolare il fattoriale
    
    Returns
    -------
    int
        Il fattoriale di n.
    
    Raises
    ------
    ValueError
        Se n è negativo.
    """
    if n < 0:
        raise ValueError("Il fattoriale non è definito per numeri negativi.")
    if n == 0:
        return 1
    else:
        return n * fattoriale(n - 1)

def conta_unici(lista):
    """
    Conta il numero di elementi unici in una lista.
    
    Parameters
    ----------
    lista : list
        La lista di elementi da analizzare.
    
    Returns
    -------
    int 
        Il numero di elementi unici nella lista.
    
    Examples
    --------
    >>> conta_unici([1,2,2,3])
    3
    """
    return len(set(lista))

def primi_fino_a_n(n)-> list[int]:
    """
    Genera una lista di numeri primi fino a n (incluso).
    
    Parameters
    ----------
    n : int
        Il limite superiore fino a cui generare numeri primi.
    
    Returns
    -------
    list
        Una lista di numeri primi fino a n.

    Examples
    --------
    >>> primi_fino_a_n(10)
    [2, 3, 5, 7]
    """
    if n < 2:
        return []
    
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for prime in primes:
            if prime * prime > num: #ottimizzazione - controllo fino a sqrt(num)
                break
            if num % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    return primes
