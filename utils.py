import sympy as sp
import numpy as np

letras_gregas = {
    "alfa": "alpha",
    "beta": "beta",
    "gama": "gamma",
    "delta": "delta",
    "épsilon": "epsilon",
    "zeta": "zeta",
    "eta": "eta",
    "teta": "theta",
    "iota": "iota",
    "capa": "kappa",
    "lambda": "lambda",
    "mi": "mu",
    "ni": "nu",
    "xi": "xi",
    "ômicron": "omicron",
    "pi": "pi",
    "ró": "rho",
    "sigma": "sigma",
    "tau": "tau",
    "upsilon": "upsilon",
    "fi": "phi",
    "chi": "chi",
    "psi": "psi",
    "ômega": "omega"
}



modelos = [
    {"Lotka-Volterra": {
        "model_description": "O modelo de Lotka-Volterra descreve a dinâmica de populações em um sistema predador-presa. Ele consiste em um sistema de equações diferenciais ordinárias (EDOs) que expressam a interação entre as populações de presas e predadores ao longo do tempo.",
        
        "params_interpretation":r""" - **$\alpha$**: Taxa de crescimento da população de presas na ausência de predadores.\n
- **$\beta$**: Taxa de predação, representando a eficiência da captura de presas pelos predadores.
- **$\gamma$**: Taxa de mortalidade dos predadores na ausência de presas.
- **$\delta$**: Taxa de conversão de presas consumidas em novos predadores.""",

        "model_behavior": "O modelo exibe oscilações periódicas na população de presas e predadores. Quando há muitas presas, a população de predadores cresce, reduzindo a população de presas. Com menos presas, os predadores diminuem, permitindo que as presas se recuperem e o ciclo recomece.",

        "model_applications": "O modelo de Lotka-Volterra é amplamente utilizado em ecologia para estudar interações predador-presa, mas também tem aplicações em economia, epidemiologia e dinâmica de sistemas biológicos.",
        "equations": ["alpha * x - beta * x * y","delta * x * y - gamma * y"],
        "variables":['x','y'],
        "initial_condition":[10.0,20.0],
        "time_span": [0, 100],
        "params": {
            "alpha": 0.25,
            "beta": 0.05,
            "gamma": 0.6,
            "delta": 0.03
        }
    }}
]



stability_colors = {
        "Atrator": "green",
        "Repulsor": "red",
        "Ponto de sela": "orange",
        "Inconclusivo": "blue",
        "Indeterminado": "gray"
    }



def find_symbolic_equilibria(equation,vars):
    """Resolve simbolicamente o sistema de equações com um tempo limite.
    Retorna uma lista vazia [] se não encontrar soluções dentro do tempo especificado."""
    try:
        solutions = sp.solve(equation, vars, dict=True, manual=True, doit=False)
        return solutions if solutions else []
    except Exception:
        return []


def jacobian_matrix(equations, vars):
    J = sp.Matrix(equations).jacobian(vars)
    return sp.simplify(J)


def geral_equation():
    x, y = sp.symbols('x y')
    f = sp.Function('f')(x, y)
    g = sp.Function('g')(x, y)
    dxdt = f
    dydt = g
    eqlatex = sp.latex(sp.Eq(sp.Symbol(r"\frac{dx}{dt}"), dxdt)) + \
        r", \quad " + sp.latex(sp.Eq(sp.Symbol(r"\frac{dy}{dt}"), dydt))
    return eqlatex


def latex_equation(equations, param_symbols,vars=None):
    """
    Converte uma lista de equações em formato LaTeX.
    
    Parâmetros:
        equations (list): Lista de strings representando as equações diferenciais.
        param_symbols (dict): Dicionário com os símbolos dos parâmetros.

    Retorna:
        str: String formatada em LaTeX com as equações organizadas no ambiente array.
    """
    all_symbols = {}
    all_symbols.update(param_symbols)  # Add parameters
    
    # Add variables to the symbol mapping
    if vars:
        for var in vars:
            all_symbols[var] = sp.Symbol(var)
    # Converter strings em expressões simbólicas de SymPy
    equations = [sp.sympify(eq, locals=all_symbols) for eq in equations]
    
    # Criar a estrutura LaTeX para exibir as equações como um sistema
      # Ambiente array para alinhar as equações à esquerda
    
    # Iterar sobre as equações e numerá-las como dx/dt, dy/dt, dz/dt, etc.
    if len(equations)>1:
        eqlatex = r"\begin{array}{l}"
        for i, eq in enumerate(equations):
            var_symbol = sp.Symbol(fr"\dfrac{{d{vars[i]}}}{{dt}}")  # Criando dxi/dt dinamicamente
            eqlatex += sp.latex(sp.Eq(var_symbol, eq)) + r" \\ \\" + "\n"  # Adicionando a equação com quebra de linha
        eqlatex += r"\end{array}"  # Fim do ambiente array

    # elif len(equations)==2:
    #     eqlatex = (
    #     r"\begin{array}{l}"  # Ambiente array com alinhamento à esquerda
    #     + sp.latex(sp.Eq(sp.Symbol(r"\dfrac{dx}{dt}"), equations[0]))  # Primeira equação
    #     + r" \\"  # Quebra de linha
    #     + r" \\"  # Quebra de linha
    #     + sp.latex(sp.Eq(sp.Symbol(r"\dfrac{dy}{dt}"), equations[1]))  # Segunda equação
    #     + r"\end{array}"  # Fim do ambiente array
    # )
    
    return eqlatex


def evaluate_equilibria(symbolic_eq, param_values,vars):
    aux_ne = [tuple([eq.get(var,0) for var in vars]) for eq in symbolic_eq]
    numerical_eq = []
    for equilibrium in aux_ne:
        aux = []
        for eq in equilibrium:
            try:
                aux.append(float(eq.subs(param_values).evalf()))
            except Exception as e:
                aux.append(float(eq))
                print(f"Erro ao avaliar a equação: {eq}")
        numerical_eq.append(aux)
    return numerical_eq


def estability_type(eigenvalues):
    if np.all(np.real(eigenvalues) < 0):
        stability = "Atrator"
    elif np.all(np.real(eigenvalues) > 0):
        stability = "Repulsor"
    elif np.any(np.real(eigenvalues) < 0) and np.any(np.real(eigenvalues) > 0):
        stability = "Ponto de sela"
    elif np.all(np.real(eigenvalues) == 0):
        stability = "Inconclusivo"
    else:
        stability = "Indeterminado"
    return stability
