"""Exceções específicas do domínio de dados de f1-ml-garage."""


class MissingColumnsError(ValueError):
    """Levantada quando um DataFrame de entrada não contém as colunas
    exigidas pela schema bruto do FastF1.

    Falhar aqui, de forma explícita e cedo, é preferível a deixar um
    ``KeyError`` genérico estourar no meio da normalização - a mensagem já
    aponta exatamente quais colunas faltam.
    """
