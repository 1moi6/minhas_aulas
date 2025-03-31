import streamlit as st
from utils import modelos, latex_equation, find_symbolic_equilibria, jacobian_matrix, evaluate_equilibria, estability_type, stability_colors,letras_gregas
import sympy as sp
import scipy.linalg as la
import numpy as np
import scipy.integrate as spi
import plotly.graph_objects as go




st.set_page_config(
    page_title="Análise qualitativa de EDOs",
    page_icon="attractor_icon.png"  # Arquivo de ícone na pasta do projeto
)

if 'modelos' not in st.session_state:
    st.session_state.modelos = modelos

def incluir_modelos(x):
    modelos = st.session_state.modelos
    if x in modelos:
        st.warning("Modelo já inserido")
        return
    st.session_state.modelos = [x,*modelos]
    st.session_state.modelo_selecionado = x
    
sb_containers = [st.sidebar.container(border=True),st.sidebar.container(border=True)]

sb_containers[0].selectbox('Selecione um modelo:',st.session_state.modelos,format_func=lambda x:list(x.keys())[0],key = 'modelo_selecionado')
sb_containers[0].radio('Selecione uma opção:',['Descrição do modelo','Análise Algébrica','Análise Numérica','Inserir seu Modelo'],key='tipo_de_analise')

if st.session_state.tipo_de_analise=="Descrição do modelo":
    st.markdown(
    "<h5 style='text-align: center;'>Informações sobre o modelo</h5>", 
    unsafe_allow_html=True)

    mdl = list(st.session_state.modelo_selecionado.values())[0]
    st.write("#### Descrição do modelo") 
    st.write(mdl['model_description']) 
    st.write("#### Equações do modelo") 
    st.write("As equações do modelo são:") 
    st.latex(latex_equation(mdl['equations'],{param.strip(): sp.Symbol(param.strip())
                         for param in list(mdl['params'].keys())},mdl['variables']))
    
    st.write("#### Interpretação dos parâmetros")
    st.markdown(mdl['params_interpretation'])
    st.write("#### Comportamento do modelo")
    st.markdown(mdl['model_behavior'])
    st.write("#### Aplicações do modelo")
    st.markdown(mdl['model_applications']) 

if st.session_state.tipo_de_analise=="Análise Algébrica":
    st.markdown(
    "<h5 style='text-align: center;'>Análise algébrica do modelo</h5>", 
    unsafe_allow_html=True)
    mdl = list(st.session_state.modelo_selecionado.values())[0]
    variables = sp.symbols(mdl['variables'])
    param_symbols = {param.strip(): sp.Symbol(param.strip()) for param in list(mdl['params'].keys())}
    eq_symbols = {**{v: sp.Symbol(v) for v in mdl['variables']}, **param_symbols}
    equations = [sp.sympify(eq, locals=eq_symbols) for eq in mdl['equations']]
    equilibria = find_symbolic_equilibria(equations, variables)
    if len(equilibria):
        # equilibria = [tuple(eq.values()) for eq in equilibria]
        st.write("#### Pontos de equilíbrio do modelo")
        st.write(
            f"Os pontos de equilíbrio são os valores de ${tuple(variables)}$ onde as equações do sistema são zeradas.")

        for i, eq in enumerate(equilibria, 1):
            st.write(f"**Ponto de equilíbrio $E_{i}$:**")
            eq_string = " "
            for v in mdl['variables']:
                eq_string += f"{sp.latex(sp.Symbol(v))} = {sp.latex(eq.get(sp.Symbol(v)))},\\quad "
            st.latex(eq_string)
        
        st.divider()
        J = jacobian_matrix(equations,variables)
        st.write("#### Matriz Jacobiana do modelo")
        st.write(
            "A matriz jacobiana é obtida a partir das derivadas parciais das equações do sistema.")
        st.latex(f"J{tuple(variables)} = {sp.latex(J)}")
        st.divider()
        
        st.write(
            "#### Matrizes jacobianas avaliadas nos pontos de equilíbrio")

    
        for i, eq in enumerate(equilibria, 1):
            J_eval = sp.simplify(J.subs(eq))
            st.markdown(
                f"Matriz Jacobiana avaliada em $E_{i}$:")
            st.latex(f"J(E_{i}) = {sp.latex(J_eval)}")

if st.session_state.tipo_de_analise=="Análise Numérica":
    st.markdown(
    "<h5 style='text-align: center;'>Análise numérica do modelo</h5>", 
    unsafe_allow_html=True
)
    mdl = list(st.session_state.modelo_selecionado.values())[0]
    variables = sp.symbols(mdl['variables'])
    param_symbols = {param.strip(): sp.Symbol(param.strip()) for param in list(mdl['params'].keys())}
    eq_symbols = {**{v: sp.Symbol(v) for v in mdl['variables']}, **param_symbols}
    equations = [sp.sympify(eq, locals=eq_symbols) for eq in mdl['equations']]
   
    equilibria = find_symbolic_equilibria(equations, variables)
    
    sb_containers[1].markdown('Intervalo de tempo para a simulação')
    time_span = mdl['time_span']
    t_0 = sb_containers[1].number_input(f"Tempo inicial: :",0.0,value=float(time_span[0]))
    t_f = sb_containers[1].number_input(f"Tempo final:",0.0,value=float(time_span[1]))
    t_span = np.linspace(t_0, t_f, 500)
    
    sb_containers[1].markdown('Condição inicial para o modelo')
    
    initial_condition = []
    for k,v in zip(variables,mdl['initial_condition']):
        aux = sb_containers[1].number_input(f"Condição inicial para ${sp.latex(k)}$:",0.0,value=v)
        initial_condition.append(aux)

    sb_containers[1].markdown('Parâmetros do modelo')
    param_values = {}
    for k,v in mdl['params'].items():
        param_values[k] = sb_containers[1].number_input(f"Valor de ${sp.latex(sp.Symbol(k))}$:",0.0,value=v)

    with st.expander("**Análise numérica dos equilíbrios**"):
        numerical_equilibria = evaluate_equilibria(equilibria, param_values,variables)
        J = jacobian_matrix(equations,variables)
        if len(numerical_equilibria):
            try:
                for i, eq in enumerate(numerical_equilibria, 1):
                    J_eval = sp.simplify(
                        J.subs({k:v for k,v in zip(variables,eq)})).subs(param_values)
                    st.write(
                        f"**{i}. Matriz Jacobiana avaliada em $E_{i}={tuple(round(e, 2) for e in eq)}$:**")
                    st.latex(f"J(E_{i}) = {sp.latex(J_eval.evalf(2))}")
                    J_numpy = np.array(J_eval.tolist(), dtype=np.float64)
                    eigenvalues = la.eigvals(J_numpy)
                    tipo = estability_type(eigenvalues)
                    st.write(f"Os autovalores são:")
                    for id,av in enumerate(eigenvalues,1):
                        symbol = f"\\lambda_{{{id}}}"
                        st.latex(f"{symbol} = {av:.2f}")
                    if "Atrator" in tipo:
                        st.success(tipo)
                    elif "sela" in tipo:
                        st.warning(tipo)
                    else:
                        st.error(tipo)
            except Exception as e:
                st.error(f"Erro ao calcular a matriz jacobiana: {e}")
    with st.expander("**Gráfico da solução numérica**"):
        numerical_function = sp.lambdify((*variables, *param_symbols.values()), equations, modules="numpy")
        def system(t, vars, param_values):
                return numerical_function(*vars, *param_values.values())
        resultado = system(0,initial_condition,param_values)

        sol = spi.solve_ivp(system, (t_span[0], t_span[-1]), initial_condition, t_eval=t_span, args=(param_values,))
        #  Criar gráfico da solução com Plotly
        fig_solucao = go.Figure()
        yaxis = []
        for idx,s in enumerate(sol.y):
            fig_solucao.add_trace(go.Scatter(x=sol.t, y=s, mode="lines", name=f"{variables[idx]}(t)"))
            yaxis.append(f"{variables[idx]}(t)")

        # Configuração do layout
        fig_solucao.update_layout(
            title="Solução numérica do modelo",
            xaxis_title="t",
            yaxis_title= ", ".join(yaxis), 
            xaxis=dict(range=[t_0, t_f]),  # Garantir que o eixo x começa em 0
            template="plotly_white"
        )
        st.plotly_chart(fig_solucao, use_container_width=True)

    if len(variables)==2:
        with st.expander("**Gráfico da plano de fase e órbita**"):
            x_vals = np.linspace(min(0, min(sol.y[0])), 1.2 * max(sol.y[0]), 20)
            y_vals = np.linspace(min(0, min(sol.y[1])), 1.2 * max(sol.y[1]), 20)
            X, Y = np.meshgrid(x_vals, y_vals)
            U, V = system(0, [X, Y], param_values)
            try:
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
                    if min(eq)>=0:
                        J_eval = sp.simplify(
                                J.subs({k:v for k,v in zip(variables,eq)})).subs(param_values)
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
                    title="Campo vetorial e órbita do sistema",
                    xaxis_title=f"{variables[0]}",  # Forçando LaTeX
                    yaxis_title=f"{variables[1]}",
                    xaxis=dict(range=[min(0, min(sol.y[0])), 1.2 * max(sol.y[0])]),
                    yaxis=dict(range=[min(0, min(sol.y[1])), 1.2 * max(sol.y[1])]), 
                    template="plotly_white"
                )
                st.plotly_chart(fig_campo, use_container_width=True)
            except:
                pass           


if st.session_state.tipo_de_analise=="Inserir seu Modelo":
    st.markdown(
    "<h5 style='text-align: center;'>Modelo personalizado</h5>", 
    unsafe_allow_html=True
)
    conts = [st.container(border=True), st.container(
                border=True), st.container(border=True)]

    conts[0].markdown("**Informe as variáveis do seu modelo, separadas por vírgula.**")
    # variables = conts[0].text_input("Variáveis do modelo:", ", ".join(f"x_{i+1}" for i in range(3)), key="variables")
    variables = conts[0].text_input("Variáveis do modelo:", ", ".join(['S','I','R']), key="variables")
    variables = [v.strip() for v in variables.split(",")]
    symbols = sp.symbols(variables)
    aux = ", ".join([sp.latex(symbol) for symbol in symbols])
    conts[0].markdown("As variáveis do modelo são: $" + aux+"$")
    conts[0].divider()
    conts[0].markdown("**Informe as equações dos seu modelo.**")
    eq_strings_a = ["-alpha*S*I","alpha*S*I-beta*I","beta*I"]
    eq_strings = []
    for i,v in enumerate(variables):
        eq_v = conts[0].text_input(
                "Digite a equação para $" + sp.latex(symbols[i]) + "$:", eq_strings_a[i], key=f"eq_{v}")
        eq_strings.append(eq_v)

    conts[0].divider()
    conts[0].markdown("**Informe os parâmetros do seu modelo, separados por vírgula.**")
    parameters = conts[0].text_input("Parâmetros do modelo:", "alpha, beta", key="parameters")
    parameters = [letras_gregas.get(v.strip(),v.strip()) for v in parameters.split(",")]
    param_symbols = sp.symbols(parameters)
    aux = ", ".join([sp.latex(symbol) for symbol in param_symbols])
    
    cols = conts[0].columns([0.75,0.25],vertical_alignment='center')
    cols[0].markdown("Os parâmetros do modelo são: $" + aux+"$")
    on = cols[1].toggle("ver equações?")
    if on:
        conts[0].write("As equações do modelo são:")
        conts[0].latex(latex_equation(eq_strings,{k:v for k,v in zip(parameters,param_symbols)},variables))
    
    conts[0].divider()
    conts[0].markdown("**Informe o valor dos parâmetros do modelo.**")
    cols = conts[0].columns(5)
    param_values = {}
    for i, p in enumerate(parameters):
        aux  = cols[i%5].number_input(f"Valor de ${sp.latex(sp.Symbol(p))}$:",0.0,value=1.0,key=f"param_{p}")
        param_values[p] = aux
    
    
    conts[0].markdown("**Informe os valores iniciais das variáveis do modelo.**")
    cols = conts[0].columns(min(5,len(variables)))
    inicial_condition = []
    for i, v in enumerate(variables):
        aux = cols[i%5].number_input(f"Valor inicial de ${sp.latex(sp.Symbol(v))}$:",0.0,value=1.0,key=f"var_{v}")
        inicial_condition.append(aux)
    
    conts[0].markdown("**Informe o intervalo de tempo para a simulação.**")
    cols = conts[0].columns(2)
    t_0 = cols[0].number_input("Tempo inicial:",0.0,value=0.0)
    t_f = cols[1].number_input("Tempo final:",0.0,value=10.0)
    

    user_model = {}
    user_model['Modelo Personalizado']= {}  
    user_model['Modelo Personalizado']["model_description"] = None
    user_model['Modelo Personalizado']["params_interpretation"] = None
    user_model['Modelo Personalizado']["model_behavior"] = None
    user_model['Modelo Personalizado']["model_applications"] = None
    user_model['Modelo Personalizado']["variables"] = variables
    user_model['Modelo Personalizado']["initial_condition"] = inicial_condition
    user_model['Modelo Personalizado']["time_span"] = [t_0,t_f]
    user_model['Modelo Personalizado']["params"] = param_values
    user_model['Modelo Personalizado']["equations"] = eq_strings

    conts[0].button("Inserir modelo",key="run_analysis",use_container_width=True,on_click=incluir_modelos,args=(user_model,))  
    