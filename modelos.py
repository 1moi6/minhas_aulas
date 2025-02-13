import streamlit as st
import sympy as sp
import numpy as np
import scipy.linalg as la
import scipy.integrate as spi
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análise qualitativa de EDOs",
    page_icon="attractor_icon.png"  # Arquivo de ícone na pasta do projeto
)

param_values = {
    sp.Symbol('r'): 0.9,
    sp.Symbol('K'): 7.0,
    sp.Symbol('alpha'): 0.02,
    sp.Symbol('s'): 1.0,
    sp.Symbol('M'): 9.0,
    sp.Symbol('beta'): 0.05
}

stability_colors = {
        "Atrator": "green",
        "Repulsor": "red",
        "Ponto de sela": "orange",
        "Inconclusivo": "blue",
        "Indeterminado": "gray"
    }


def find_symbolic_equilibria(eq1, eq2, vars):
    return sp.solve([eq1, eq2], vars)


def jacobian_matrix(eq1, eq2, vars):
    J = sp.Matrix([eq1, eq2]).jacobian(vars)
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


def latex_equation(eq1, eq2):
    eqlatex = (
        r"\begin{array}{l}"  # Ambiente array com alinhamento à esquerda
        + sp.latex(sp.Eq(sp.Symbol(r"\dfrac{dx}{dt}"), eq1))  # Primeira equação
        + r" \\"  # Quebra de linha
        + r" \\"  # Quebra de linha
        + sp.latex(sp.Eq(sp.Symbol(r"\dfrac{dy}{dt}"), eq2))  # Segunda equação
        + r"\end{array}"  # Fim do ambiente array
    )
    return eqlatex


def evaluate_equilibria(symbolic_eq, param_values):
    numerical_eq = [
        tuple(float(eq.subs(param_values).evalf()) for eq in equilibrium)
        for equilibrium in symbolic_eq
    ]
    return numerical_eq


def mudapar2(chave, valor):
    st.session_state[chave] = valor
    st.session_state[f"tab1_{chave}"] = valor
    st.rerun()


def mudapar1(chave, valor):
    st.session_state[chave] = valor
    st.session_state[f"tab2_{chave}"] = valor
    st.rerun()


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

def main():
    st.write("#### Análise qualitativa de equações diferenciais")
    tabs = st.tabs(
        ["Equações e parâmetros", 'Análise de estabilidade e solução'])

    with tabs[0]:
        conts = [st.container(border=True), st.container(
            border=True), st.container(border=True)]
        conts[0].write("#### Equações e parâmetros")
        conts[0].write(
            "Digite as equações diferenciais do sistema dinâmico e os parâmetros utilizados. A equação geral do sistema é:")
        conts[0].latex(geral_equation())

        x, y = sp.symbols('x y')
        conts[0].divider()
        conts[0].write(
            "\nInsira aqui todos os parâmetros utilizados no seu modelo. Os parâmetros devem ser informados separados por vírgulas.")
        params_input = conts[0].text_input(
            "**Digite todos os parâmetros separados por vírgulas:**", "r, K, alpha, s, M, beta")
        param_symbols = {param.strip(): sp.Symbol(param.strip())
                         for param in params_input.split(',') if param.strip()}

        conts[0].write("Seus parâmetros são: " +
                       ", ".join([f"${sp.latex(v)}$" for v in param_symbols.values()]))

        conts[0].divider()
        eq1_input = conts[0].text_input(
            "Digite a função $f(x,y)$", "r * x * (1 - x / K) + alpha * x * y", key="eq1")
        conts[0].write(
            "\nEsta função representa a taxa de variação de $x$ ao longo do tempo.")
        eq2_input = conts[0].text_input(
            "Digite a função $g(x,y)$", "s * y * (1 - y / M) + beta * x * y", key="eq2")
        conts[0].write(
            "\nEsta função representa a taxa de variação de $y$ ao longo do tempo.")

        try:
            eq1 = sp.sympify(eq1_input, locals=param_symbols)
            eq2 = sp.sympify(eq2_input, locals=param_symbols)
            conts[0].write("\nAs equações do sistema são:")
            conts[0].latex(latex_equation(eq1, eq2))

            equilibria = find_symbolic_equilibria(eq1, eq2, (x, y))
            J = jacobian_matrix(eq1, eq2, (x, y))
            cols_pe = conts[1].columns([0.6,0.4])
            cols_pe[0].write("#### Pontos de equilíbrio")
            if cols_pe[1].toggle("Ver pontos de equilíbrio",value=True):
                conts[1].write(
                    "Os pontos de equilíbrio são os valores de $(x,y)$ onde as equações do sistema são zeradas.")

                for i, eq in enumerate(equilibria, 1):
                    conts[1].write(f"Ponto de equilíbrio {i}:")
                    conts[1].latex(
                        f"E_{i}=\\left({sp.latex(eq[0])}, {sp.latex(eq[1])}\\right)")
            
            cols_ja = conts[2].columns([0.7,0.3])
            cols_ja[0].write("#### Matriz jacobiana do sistema")
            if cols_ja[1].toggle("Ver jacobiana",value=True):
                conts[2].write(
                    "A matriz jacobiana é obtida a partir das derivadas parciais das equações do sistema.")
                conts[2].latex(f"J(x,y) = {sp.latex(J)}")
                conts[2].divider()
                conts[2].write(
                    "##### Matrizes jacobianas avaliadas nos pontos de equilíbrio")

                for i, eq in enumerate(equilibria, 1):
                    J_eval = sp.simplify(J.subs({x: eq[0], y: eq[1]}))
                    conts[2].write(
                        f"Matriz Jacobiana avaliada no ponto de equilíbrio {i}:")
                    conts[2].latex(f"J(E_{i}) = {sp.latex(J_eval)}")

        except Exception as e:
            st.error(f"Erro ao processar as equações: {e}")

    with tabs[1]:
        conts1 = [st.container(border=True), st.container(
            border=True), st.container(border=True), st.container(border=True)]
        conts1[0].write("#### Definição dos parâmetros")
        cols = conts1[0].columns([0.2, 0.2, 0.2, 0.2, 0.2])
        # param_values = {}
        for id, k in enumerate(param_symbols.values()):
            pos = id % 5
            param_values[k] = cols[pos].number_input(
                f"Valor de ${sp.latex(k)}$:",
                0.0, None, param_values.get(k,1.0)
            )

        numerical_equilibria = evaluate_equilibria(equilibria, param_values)
        col1 = conts1[1].columns([0.8, 0.2])
        on = col1[1].toggle("Ver análise")
        col1[0].write("#### Avaliação dos equilíbrios")
        if on:
            for i, eq in enumerate(numerical_equilibria, 1):
                J_eval = sp.simplify(
                    J.subs({x: eq[0], y: eq[1]})).subs(param_values)
                conts1[1].write(
                    f"**{i}. Matriz Jacobiana avaliada no ponto de equilíbrio $E_{i}=({eq[0]:.2f},{eq[1]:.2f})$:**")
                conts1[1].latex(f"J(E_{i}) = {sp.latex(J_eval.evalf(3))}")
                J_numpy = np.array(J_eval.tolist(), dtype=np.float64)
                eigenvalues = la.eigvals(J_numpy)
                tipo = estability_type(eigenvalues)
                conts1[1].write(
                    f"Os autovalores são: $\\lambda_{1} = {eigenvalues[0]:.3f}$ e $\\lambda_{2} = {eigenvalues[1]:.3f}$")
                if "Atrator" in tipo:
                    conts1[1].success(tipo)
                elif "sela" in tipo:
                    conts1[1].warning(tipo)
                else:
                    conts1[1].error(tipo)

        col2 = conts1[2].columns([0.8, 0.2])
        on2 = col2[1].toggle("Ver gráficos")
        col2[0].write("#### Plano de fase e solução")
        cols = conts1[2].columns([1/3, 1/3, 1/3])

        if on2:
            # Entrada dos parâmetros iniciais
            x0 = cols[0].number_input(
                f"Valor inicial de $x(t)$:",
                0.0, None, 1.0
            )
            y0 = cols[1].number_input(
                f"Valor inicial de $y(t)$:",
                0.0, None, 1.0
            )
            tf = cols[2].number_input(
                f"Intervalo de tempo da solução:",
                0.0, None, 20.0
            )

            # Gerar lista de parâmetros
            param_list = list(param_symbols.values())
            print(param_list)

            # Criar funções numéricas
            f_lambda = sp.lambdify((x, y, *param_list), eq1, modules="numpy")
            g_lambda = sp.lambdify((x, y, *param_list), eq2, modules="numpy")

            # Configuração do solver
            y0 = [x0, y0]  # Condições iniciais
            t_span = np.linspace(0, tf, 500)  

            # Definir a função do sistema dinâmico
            def system(t, vars, param_values):
                x_val, y_val = vars
                return [
                    f_lambda(x_val, y_val, *param_values.values()),
                    g_lambda(x_val, y_val, *param_values.values()),
                ]

            # Resolver as equações diferenciais
            par_vals = {key:val for key, val in param_values.items() if key in param_symbols.values()}
            sol = spi.solve_ivp(system, (t_span[0], t_span[-1]), y0, t_eval=t_span, args=(par_vals,))

            # Criar gráfico da solução com Plotly
            fig_solucao = go.Figure()
            fig_solucao.add_trace(go.Scatter(x=sol.t, y=sol.y[0], mode="lines", name="x(t)"))
            fig_solucao.add_trace(go.Scatter(x=sol.t, y=sol.y[1], mode="lines", name="y(t)"))

            # Configuração do layout
            fig_solucao.update_layout(
                title="Solução do sistema de EDOs",
                xaxis_title="Tempo",
                yaxis_title="População",
                xaxis=dict(range=[0, tf]),  # Garantir que o eixo x começa em 0
                template="plotly_white"
            )

            # Exibir gráfico interativo
            conts1[2].plotly_chart(fig_solucao, use_container_width=True)

            # Definir os valores para o campo vetorial
            x_vals = np.linspace(min(0, min(sol.y[0])), 1.2 * max(sol.y[0]), 20)
            y_vals = np.linspace(min(0, min(sol.y[1])), 1.2 * max(sol.y[1]), 20)
            X, Y = np.meshgrid(x_vals, y_vals)
            U, V = system(0, [X, Y], par_vals)

            # Criar campo vetorial usando Plotly
            fig_campo = go.Figure()

            # Adicionar setas do campo vetorial
            for i in range(len(X)):
                for j in range(len(Y)):
                    fig_campo.add_trace(go.Scatter(
                        x=[X[i, j], X[i, j] + U[i, j] * 0.1],
                        y=[Y[i, j], Y[i, j] + V[i, j] * 0.1],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        showlegend=False
                    ))

            # Adicionar trajetória da solução
            fig_campo.add_trace(go.Scatter(
                x=sol.y[0], y=sol.y[1], mode="lines", name="Trajetória", line=dict(color="blue"),showlegend=False
            ))

            # Adicionar pontos de equilíbrio
            used_labels = set()
            for i, eq in enumerate(numerical_equilibria, 1):
                J_eval = sp.simplify(J.subs({x: eq[0], y: eq[1]})).subs(param_values)
                J_numpy = np.array(J_eval.tolist(), dtype=np.float64)
                eigenvalues = la.eigvals(J_numpy)
                tipo = estability_type(eigenvalues)
                show_legend = tipo not in used_labels  
                used_labels.add(tipo) 
                
                fig_campo.add_trace(go.Scatter(
                    x=[eq[0]], y=[eq[1]],
                    mode="markers",
                    marker=dict(size=10, color=stability_colors[tipo]),
                    name=tipo if show_legend else "",
                    showlegend=show_legend
                ))

            # Configuração do layout do campo vetorial
            fig_campo.update_layout(
                title="Campo vetorial",
                xaxis_title="x",  # Forçando LaTeX
                yaxis_title="y", 
                template="plotly_white"
            )

            # Exibir campo vetorial interativo
            conts1[2].plotly_chart(fig_campo, use_container_width=True)
if __name__ == "__main__":
    main()
