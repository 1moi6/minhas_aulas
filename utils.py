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
    }},

{
    "Holling-Tanner": {
        "model_description": "O modelo de Holling-Tanner é uma extensão do modelo predador-presa que incorpora uma resposta funcional tipo II para os predadores e crescimento logístico para a população de presas. Isso torna o modelo mais realista em termos ecológicos, ao considerar limitações ambientais para as presas e saturação na taxa de predação.",
        
        "params_interpretation": r"""- **$r$**: Taxa intrínseca de crescimento da população de presas.\n
- **$K$**: Capacidade de suporte ambiental da população de presas.\n
- **$a$**: Taxa máxima de predação por predador.\n
- **$b$**: Constante de saturação na resposta funcional do tipo II (quanto maior, menor a eficiência da predação em altas densidades de presas).\n
- **$d$**: Taxa de mortalidade natural dos predadores.\n
- **$c$**: Taxa de conversão de presas consumidas em novos predadores.""",

        "model_behavior": "O modelo pode exibir diferentes dinâmicas, incluindo pontos de equilíbrio estáveis e oscilações amortecidas, dependendo dos parâmetros. A introdução da capacidade de suporte limita o crescimento indefinido das presas, e a saturação na taxa de predação impede que os predadores eliminem completamente as presas.",
        
        "model_applications": "É utilizado em ecologia para descrever interações predador-presa de forma mais realista, especialmente quando há limitação ambiental para as presas ou saturação na eficiência de predação. Também pode ser aplicado em epidemiologia e sistemas biológicos com competição ou cooperação assimétrica.",
        
        "equations": ["r * x * (1 - x / K) - (a * x * y / (1 + h*x))", "s*y*(1-y/(M*x))"],
        
        "variables": ["x", "y"],
        
        "initial_condition": [5.0, 5.0],
        
        "time_span": [0.0, 100.0],
        
        "params": {
            "r": 1.2,
            "K": 50.0,
            "a": 0.5,
            "h": 0.1,
            "s": 0.2,
            "M": 2.0
        }
    }
},

{
    "Modelo SIR": {
        "model_description": "O modelo SIR descreve a propagação de uma doença infecciosa em uma população dividida em três compartimentos: suscetíveis (S), infectados (I) e recuperados (R). É um sistema de equações diferenciais ordinárias (EDOs) que captura como indivíduos transitam entre esses estados ao longo do tempo.",
        
        "params_interpretation": r"""- **$\beta$**: Taxa de transmissão da doença, que representa a probabilidade de contato entre suscetíveis e infectados resultar em infecção.\n
- **$\gamma$**: Taxa de recuperação, ou seja, a fração de infectados que se recupera por unidade de tempo.\n
- **$N$**: Tamanho total da população (geralmente considerado constante, $N = S + I + R$).""",

        "model_behavior": "O modelo pode exibir um surto epidêmico, onde a população infectada cresce rapidamente até atingir um pico, seguido de um declínio à medida que os indivíduos se recuperam e a população suscetível diminui. Se a taxa de transmissão for suficientemente alta, pode haver uma epidemia significativa.",
        
        "model_applications": "O modelo SIR é amplamente utilizado em epidemiologia para prever a evolução de surtos e epidemias, avaliar estratégias de vacinação e entender a dinâmica de doenças infecciosas em populações.",
        
        "equations": [
            "-beta * S * I / N",
            "beta * S * I / N - gamma * I",
            "gamma * I"
        ],
        
        "variables": ["S", "I", "R"],
        
        "initial_condition": [990.0, 10.0, 0.0],
        
        "time_span": [0, 160],
        
        "params": {
            "beta": 0.3,
            "gamma": 0.1,
            "N": 1000.0
        }
    }
},
{
    "Modelo logístico": {
        "model_description": "O modelo logístico descreve o crescimento de uma população em um ambiente com recursos limitados. Ao contrário do modelo exponencial, ele considera uma capacidade de suporte ambiental que limita o crescimento à medida que a população se aproxima de um valor máximo sustentável.",
        
        "params_interpretation": r"""- **$r$**: Taxa intrínseca de crescimento da população.\n
- **$K$**: Capacidade de suporte do ambiente, ou seja, o tamanho máximo da população que o ambiente pode sustentar indefinidamente.""",
        
        "model_behavior": "O modelo apresenta um crescimento rápido inicialmente (quase exponencial), mas esse crescimento diminui à medida que a população se aproxima da capacidade de suporte, estabilizando-se em torno de $K$. É um modelo clássico de crescimento limitado.",
        
        "model_applications": "Muito utilizado em ecologia para modelar crescimento populacional, o modelo logístico também aparece em contextos como difusão de inovações, dinâmica de recursos renováveis, economia e sociologia.",
        
        "equations": ["r * P * (1 - P / K)"],
        
        "variables": ["P"],
        
        "initial_condition": [10.0],
        
        "time_span": [0, 50],
        
        "params": {
            "r": 0.2,
            "K": 100.0
        }
    }
},

{
    "SIR com acumulado Z": {
        "model_description": "Essa versão do modelo SIR reformula o sistema usando a variável acumulada de casos $Z = I + R$. Isso permite expressar a dinâmica apenas em termos de $I$ (infectados) e $Z$ (casos acumulados), considerando que $S = P - Z$, onde $P$ é a população total. A equação de $I$ representa o número de indivíduos atualmente infectados, enquanto $Z$ representa todos os que foram infectados até o momento (tanto os ainda doentes quanto os recuperados).",
        
        "params_interpretation": r"""- **$\beta$**: Taxa de transmissão da doença, controlando o número de novos infectados por contato entre suscetíveis e infectados.\n
- **$\gamma$**: Taxa de recuperação dos infectados.\n
- **$P$**: População total, considerada constante (isto é, $S + I + R = P$ em todos os tempos).""",
        
        "model_behavior": "Esse modelo mantém o comportamento qualitativo do modelo SIR: o número de infectados pode crescer rapidamente se a taxa de transmissão for alta, atingindo um pico e depois diminuindo à medida que mais indivíduos passam a integrar o grupo de recuperados. O uso da variável $Z$ permite acompanhar diretamente o número acumulado de casos, o que é útil para análise de dados epidemiológicos reais.",
        
        "model_applications": "É utilizado para descrever a evolução de doenças infecciosas considerando dados acumulados de casos (Z). Isso facilita a calibração com dados empíricos e é especialmente útil em contextos de vigilância epidemiológica e modelagem preditiva.",
        
        "equations": [
            "beta * (P - Z) * I / P - gamma * I",  # dI/dt
            "beta * (P - Z) * I / P"              # dZ/dt
        ],
        
        "variables": ["I", "Z"],
        
        "initial_condition": [10.0, 10.0],
        
        "time_span": [0, 160],
        
        "params": {
            "beta": 0.3,
            "gamma": 0.1,
            "P": 1000.0
        }
    }
},

{
    "RPS ecológico": {
        "model_description": "Este modelo descreve interações cíclicas entre três tipos ou estratégias, como no clássico jogo pedra-papel-tesoura. Um exemplo notável ocorre entre lagartos do gênero *Uta*, onde três morfologias competem entre si em um ciclo evolutivo estável. O modelo usa equações diferenciais para representar como a frequência de cada tipo muda ao longo do tempo.",
        
        "params_interpretation": r"""- **$a$**: Intensidade da competição assimétrica entre os tipos.\n
- O modelo assume que as populações competem por proporções relativas (frequências) e que a soma das três populações é constante.""",
        
        "model_behavior": "O sistema exibe oscilações cíclicas contínuas ou quase periódicas entre os três tipos. Cada tipo domina temporariamente o sistema, mas logo é superado por outro. Esse comportamento é típico de sistemas com interações não transituvas e competição cíclica.",
        
        "model_applications": "Esse tipo de modelo é usado em ecologia evolutiva, teoria dos jogos, biologia de populações e até em dinâmicas sociais. É útil para estudar coexistência cíclica de estratégias ou espécies em competição.",
        
        "equations": [
            "x * (a * z - a * y)",  # dx/dt
            "y * (a * x - a * z)",  # dy/dt
            "z * (a * y - a * x)"   # dz/dt
        ],
        
        "variables": ["x", "y", "z"],
        
        "initial_condition": [0.33, 0.33, 0.34],
        
        "time_span": [0, 100],
        
        "params": {
            "a": 1.0
        }
    }
},

{
    "Competição Lotka-Volterra": {
        "model_description": "Esse modelo descreve a dinâmica de duas espécies que competem por recursos limitados. Ao contrário do modelo predador-presa, ambas as espécies prejudicam o crescimento uma da outra. O modelo avalia as possibilidades de coexistência ou exclusão competitiva dependendo da intensidade da competição interespécies.",
        
        "params_interpretation": r"""- **$r_1$**, **$r_2$**: Taxas de crescimento das espécies 1 e 2, respectivamente.\n
- **$K_1$**, **$K_2$**: Capacidades de suporte do ambiente para cada espécie na ausência da outra.\n
- **$\alpha$**: Efeito da espécie 2 sobre a espécie 1.\n
- **$\beta$**: Efeito da espécie 1 sobre a espécie 2.""",
        
        "model_behavior": "O sistema pode evoluir para diferentes equilíbrios: uma das espécies pode ser excluída, ambas podem coexistir de forma estável, ou o sistema pode ser sensível às condições iniciais. O resultado depende das relações entre os coeficientes de competição e as capacidades de suporte.",
        
        "model_applications": "É utilizado para estudar competição entre espécies na ecologia, inclusive em estratégias de controle biológico ou em dinâmica populacional em nichos sobrepostos.",
        
        "equations": [
            "r1 * x * (1 - (x + alpha * y) / K1)",
            "r2 * y * (1 - (y + beta * x) / K2)"
        ],
        
        "variables": ["x", "y"],
        
        "initial_condition": [10.0, 10.0],
        
        "time_span": [0, 100],
        
        "params": {
            "r1": 0.5,
            "r2": 0.4,
            "K1": 50.0,
            "K2": 60.0,
            "alpha": 0.6,
            "beta": 0.7
        }
    }
},

{
    "Modelo SIRS": {
        "model_description": "O modelo SIRS é uma extensão do modelo SIR que inclui a perda de imunidade. Após a recuperação, os indivíduos eventualmente voltam ao grupo suscetível. Isso é adequado para doenças em que a imunidade é temporária.",
        
        "params_interpretation": r"""- **$\beta$**: Taxa de transmissão da doença.\n
- **$\gamma$**: Taxa de recuperação.\n
- **$\xi$**: Taxa de perda de imunidade (ou reinfecção).\n
- **$N$**: População total.""",
        
        "model_behavior": "Diferentemente do modelo SIR, aqui a epidemia pode se tornar endêmica, com ciclos repetitivos de infecção. O número de infectados pode estabilizar ou oscilar de acordo com os parâmetros.",
        
        "model_applications": "Adequado para modelar doenças com imunidade temporária, como gripe, dengue ou outras doenças sazonais. Também usado para estudar vacinação recorrente ou imunidade de curta duração.",
        
        "equations": [
            "-beta * S * I / N + xi * R",
            "beta * S * I / N - gamma * I",
            "gamma * I - xi * R"
        ],
        
        "variables": ["S", "I", "R"],
        
        "initial_condition": [990.0, 10.0, 0.0],
        
        "time_span": [0, 160],
        
        "params": {
            "beta": 0.3,
            "gamma": 0.1,
            "xi": 0.05,
            "N": 1000.0
        }
    }
}
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
    else:
        var_symbol = sp.Symbol(fr"\dfrac{{d{vars[0]}}}{{dt}}") 
        eqlatex = sp.latex(sp.Eq(var_symbol, equations[0]))
    
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
                try:
                    aux.append(float(eq))
                except:
                    aux.append(0)

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
