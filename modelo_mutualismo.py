import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.integrate as spi
import sympy as sp

# Função do sistema dinâmico

def find_symbolic_equilibria():
    x, y, r, K, alpha, s, M, beta = sp.symbols('x y r K alpha s M beta')
    dxdt = r * x * (1 - x / K) + alpha * x * y
    dydt = s * y * (1 - y / M) + beta * x * y
    return sp.solve([dxdt, dydt], (x, y))

def latex_equation():
    # Definição dos símbolos
    x, y, r, K, alpha, s, M, beta = sp.symbols('x y r K alpha s M beta')

    # Definição das equações diferenciais
    dxdt = r * x * (1 - x / K) + alpha * x * y
    dydt = s * y * (1 - y / M) + beta * x * y

    # Converter para LaTeX
    eqlatex = sp.latex(sp.Eq(sp.Symbol(r"\frac{dx}{dt}"), dxdt)) + r", \quad " + sp.latex(sp.Eq(sp.Symbol(r"\frac{dy}{dt}"), dydt))

    return eqlatex

symbolic_equilibria = find_symbolic_equilibria()
def evaluate_equilibria(symbolic_eq, param_values):
    numerical_eq = [
    tuple(float(eq.subs(param_values).evalf()) for eq in equilibrium)
    for equilibrium in symbolic_eq
]
    return numerical_eq

def system(t, vars, r, K, alpha, s, M, beta):
    x, y = vars
    dxdt = r * x * (1 - x / K) + alpha * x * y
    dydt = s * y * (1 - y / M) + beta * x * y
    return [dxdt, dydt]

# Função para calcular pontos de equilíbrio
def find_equilibria(r, K, alpha, s, M, beta):
    equilibria = [(0, 0), (K, 0), (0, M)]
    if alpha != 0 and beta != 0:
        denom_x = s - beta * M - alpha * K
        denom_y = r - alpha * K - beta * M

        if denom_x != 0 and denom_y != 0:
            eq_x = K * (s - beta * M) / denom_x
            eq_y = M * (r - alpha * K) / denom_y
            equilibria.append((eq_x, eq_y))

    return equilibria

# Função para calcular a matriz Jacobiana
def jacobian_matrix(x, y, r, K, alpha, s, M, beta):
    J = np.array([
        [r * (1 - 2*x/K) + alpha * y, alpha * x],
        [beta * y, s * (1 - 2*y/M) + beta * x]
    ])
    return J

# Interface no Streamlit
st.title("Análise Qualitativa de EDOs")

# Sidebar para selecionar parâmetros
st.sidebar.header("Parâmetros")
r = st.sidebar.slider("r", 0.01, 3.0, 0.9)
K = st.sidebar.slider("K", 0.01, 10.0, 5.5)
alpha = st.sidebar.slider("alpha", 0.0, 0.2, 0.02)
s = st.sidebar.slider("s", 0.01, 2.0, 1.0)
M = st.sidebar.slider("M", 0.1, 10.0, 7.2)
beta = st.sidebar.slider("beta", 0.0, 0.5, 0.05)

# Sidebar para condições iniciais e tempo
t_max = st.sidebar.slider("Tempo máximo", 10, 50, 10)
x0 = st.sidebar.slider("x0", 0.0, 3*K, K/20)
y0 = st.sidebar.slider("y0", 0.0, 3*M, 2*M)

param_values = {
    sp.Symbol('r'): r,
    sp.Symbol('K'): K,
    sp.Symbol('alpha'): alpha,
    sp.Symbol('s'): s,
    sp.Symbol('M'): M,
    sp.Symbol('beta'): beta
}
numerical_equilibria = evaluate_equilibria(symbolic_equilibria, param_values)
# Tabs para organização
tabs = st.tabs(["O modelo","Equilíbrio", "Estabilidade", "Plano de fases", "Trajetórias no tempo"])

with tabs[0]:
    st.write("#### Sistema de Equações Diferenciais")

    # Exibir as equações diferenciais formatadas em LaTeX
    st.latex(latex_equation())

    # Explicações biológicas e hipóteses do modelo
    
    st.markdown("""
    O modelo matemático representa a **dinâmica de duas espécies mutualistas**, onde cada uma auxilia no crescimento da outra. Para sua construção, foram consideradas as seguintes hipóteses:

    - **Crescimento intrínseco**: Cada espécie cresce de maneira logística, limitada por sua **capacidade suporte** na ausência da outra.
    - **Interação mutualística**: Ocorre um **benefício mútuo**, onde a presença de uma espécie **aumenta a taxa de crescimento** da outra, proporcionalmente aos coeficientes de mutualismo.
    - **capacidade suporte**: Cada espécie tem um limite máximo de indivíduos sustentado pelo ambiente, caso não receba apoio da outra espécie.
    - **Ausência de predação ou competição direta**: O modelo considera apenas interações **positivas**, sem predação ou competição entre as espécies.
    - **Interação proporcional às populações**: Quanto maior a população de uma espécie, maior o benefício gerado para a outra.
    """)


    # Explicação Biológica com Melhor Formatação
    



with tabs[1]:
    st.write("#### Pontos de equilíbrio do modelo")
    explanations = [
        ("1. Extinção Total", 
         "Ambas as populações se extinguem, o que pode ocorrer se a taxa de crescimento for muito baixa ou se o impacto negativo da interação for muito alto.",
         symbolic_equilibria[0]),

        ("2. Sobrevivência Apenas da Espécie 1", 
         "A população da espécie 1 atinge sua capacidade suporte \( K \), enquanto a espécie 2 é extinta. Isso pode ocorrer se a segunda espécie for muito dependente da primeira e não conseguir se sustentar sozinha.",
         symbolic_equilibria[1]),

        ("3. Sobrevivência Apenas da Espécie 2", 
         "A população da espécie 2 atinge sua capacidade suporte \( M \), enquanto a espécie 1 é extinta. Esse caso é análogo ao anterior, mas favorece a segunda espécie.",
         symbolic_equilibria[2]),

        ("4. Coexistência das Espécies", 
         "As duas espécies atingem um equilíbrio populacional positivo e coexistem. Esse é o cenário de mutualismo esperado, onde ambas as espécies se beneficiam da interação.",
         symbolic_equilibria[3] if len(symbolic_equilibria) > 3 else None)
    ]

    # Exibir cada ponto de equilíbrio junto com sua explicação
    for title, description, eq in explanations:
        st.write(f"##### {title}")
        st.write(description)
        
        if eq is not None:
            st.latex(f"\\left( {sp.latex(eq[0])}, {sp.latex(eq[1])} \\right)")
        else:
            st.write("*Esse equilíbrio não existe para os parâmetros atuais.*")


    
with tabs[2]:
    st.write(f"#### Pontos de equilíbrio e estabilidade")
    equilibria = numerical_equilibria
    formatted_equilibria = [(f"({x:.2f}, {y:.2f})", (x, y)) for x, y in numerical_equilibria]

# Criar dropdown para seleção do equilíbrio
    selected_label, selected_eq = st.selectbox("Selecione um ponto de equilíbrio:", formatted_equilibria, format_func=lambda x: x[0])

    # Separando coordenadas do equilíbrio selecionado
    x_eq, y_eq = selected_eq
    J = jacobian_matrix(x_eq, y_eq, r, K, alpha, s, M, beta)
    eigenvalues = la.eigvals(J)

    # Determinação da estabilidade
    if np.all(np.real(eigenvalues) < 0):
        stability = "Estável (nó atrator)"
    elif np.all(np.real(eigenvalues) > 0):
        stability = "Instável (nó repulsor)"
    elif np.any(np.real(eigenvalues) < 0) and np.any(np.real(eigenvalues) > 0):
        stability = "Ponto de sela"
    elif np.all(np.real(eigenvalues) == 0):
        stability = "Caso marginal (mais informações necessárias)"
    else:
        stability = "Indeterminado"

    # Exibir ponto de equilíbrio numérico formatado
    # Exibir matriz Jacobiana formatada em LaTeX
    st.write("##### Matriz Jacobiana no Equilíbrio")
    st.latex(
        f"J = \\begin{{bmatrix}} {J[0,0]:.3f} & {J[0,1]:.3f} \\\\ {J[1,0]:.3f} & {J[1,1]:.3f} \\end{{bmatrix}}"
    )

    # Exibir autovalores formatados
    st.write("##### Autovalores da Matriz Jacobiana")
    eigen_str = " \\quad ".join([f"{eig:.3f}" for eig in eigenvalues])
    st.latex(f"\\lambda_1, \\lambda_2 = {eigen_str}")

    # Exibir tipo de estabilidade
    st.write("##### Tipo de Estabilidade")
    if "Estável" in stability:
        st.success(stability)
    elif "sela" in stability:
        st.warning(stability)
    else:
        st.error(stability)

with tabs[3]:  # Corrigindo o índice para corresponder à aba correta
    st.write("#### Plano de fases")
    
    # Definir os valores para o campo vetorial
    x_vals = np.linspace(0, 2*K, 20)
    y_vals = np.linspace(0, 2*M, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    U, V = system(0, [X, Y], r, K, alpha, s, M, beta)

    # Criar figura e eixo para o plano de fases
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, color="gray", alpha=0.6)  # Campo vetorial
    ax.set_xlabel("$x$ (Espécie 1)")
    ax.set_ylabel("$y$ (Espécie 2)")
    ax.set_title("Campo vetorial")

    # Cores para diferentes tipos de estabilidade
    stability_colors = {
        "Estável (nó atrator)": "green",
        "Instável (nó repulsor)": "red",
        "Ponto de sela": "orange",
        "Caso marginal (mais informações necessárias)": "blue",
        "Indeterminado": "gray"
    }

    # Adicionar os pontos de equilíbrio com cores de acordo com a estabilidade
    for eq in numerical_equilibria:
        x_eq, y_eq = eq
        J = jacobian_matrix(x_eq, y_eq, r, K, alpha, s, M, beta)
        eigenvalues = la.eigvals(J)

        # Determinar estabilidade do ponto
        if np.all(np.real(eigenvalues) < 0):
            stability = "Estável (nó atrator)"
        elif np.all(np.real(eigenvalues) > 0):
            stability = "Instável (nó repulsor)"
        elif np.any(np.real(eigenvalues) < 0) and np.any(np.real(eigenvalues) > 0):
            stability = "Ponto de sela"
        elif np.all(np.real(eigenvalues) == 0):
            stability = "Caso marginal (mais informações necessárias)"
        else:
            stability = "Indeterminado"

        # Plottar ponto de equilíbrio com cor apropriada
        ax.plot(x_eq, y_eq, "o", color=stability_colors[stability], markersize=8, label=stability)

    # Trajetória da solução com condições iniciais
    t_eval = np.linspace(0, t_max, 1000)
    sol = spi.solve_ivp(system, [0, t_max], [x0, y0], args=(r, K, alpha, s, M, beta), t_eval=t_eval)
    ax.plot(sol.y[0], sol.y[1], "b", label="Trajetória")

    # Evitar duplicação da legenda
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

    # Exibir a legenda da estabilidade no Streamlit
    st.write("##### Tipo de estabilidade dos equilíbrios")
    for eq in numerical_equilibria:
        x_eq, y_eq = eq
        J = jacobian_matrix(x_eq, y_eq, r, K, alpha, s, M, beta)
        eigenvalues = la.eigvals(J)

        # Determinar estabilidade novamente
        if np.all(np.real(eigenvalues) < 0):
            stability = "Estável (nó atrator)"
            st.success(f"Equilíbrio ({x_eq:.2f}, {y_eq:.2f}): {stability}")
        elif np.all(np.real(eigenvalues) > 0):
            stability = "Instável (nó repulsor)"
            st.error(f"Equilíbrio ({x_eq:.2f}, {y_eq:.2f}): {stability}")
        elif np.any(np.real(eigenvalues) < 0) and np.any(np.real(eigenvalues) > 0):
            stability = "Ponto de sela"
            st.warning(f"Equilíbrio ({x_eq:.2f}, {y_eq:.2f}): {stability}")
        elif np.all(np.real(eigenvalues) == 0):
            stability = "Caso marginal (mais informações necessárias)"
            st.info(f"Equilíbrio ({x_eq:.2f}, {y_eq:.2f}): {stability}")
        else:
            stability = "Indeterminado"
            st.write(f"Equilíbrio ({x_eq:.2f}, {y_eq:.2f}): {stability}")


with tabs[4]:  # Corrigindo o índice para corresponder corretamente à aba
    st.write("#### Evolução temporal das espécies")

    # Resolver o sistema dinâmico
    sol = spi.solve_ivp(system, [0, t_max], [x0, y0], args=(r, K, alpha, s, M, beta), t_eval=t_eval)
    
    # Criar gráfico
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label="x (população 1)", color="blue")
    ax.plot(sol.t, sol.y[1], label="y (população 2)", color="red")

    # Ajustar eixos
    ax.set_xlim(left=0)  # Garante que o eixo X começa em zero
    ax.set_xlabel("Tempo")
    ax.set_ylabel("População")
    ax.set_title("Trajetórias Temporais")
    ax.legend()

    # Exibir no Streamlit
    st.pyplot(fig)

