import streamlit as st
from utils import modelos, latex_equation, find_symbolic_equilibria, jacobian_matrix, evaluate_equilibria
import sympy as sp


st.set_page_config(
    page_title="Análise qualitativa de EDOs",
    page_icon="attractor_icon.png"  # Arquivo de ícone na pasta do projeto
)


sb_containers = [st.sidebar.container(border=True),st.sidebar.container(border=True)]

sb_containers[0].selectbox('Selecione um modelo:',modelos,format_func=lambda x:list(x.keys())[0],key = 'modelo_selecionado')
sb_containers[0].radio('Selecione uma opção:',['Descrição do modelo','Análise Algébrica','Análise Numérica'],key='tipo_de_analise')

if st.session_state.tipo_de_analise=="Descrição do modelo":
    mdl = list(st.session_state.modelo_selecionado.values())[0]
    st.write("#### Descrição do modelo") 
    st.write(mdl['model_description']) 
    st.write("#### Equações do modelo") 
    st.latex(latex_equation(mdl['equations'],{param.strip(): sp.Symbol(param.strip())
                         for param in list(mdl['params'].keys())}))
    
    st.write("#### Interpretação dos parâmetros")
    st.markdown(mdl['params_interpretation'])
    st.write("#### Comportamento do modelo")
    st.markdown(mdl['model_behavior'])
    st.write("#### Aplicações do modelo")
    st.markdown(mdl['model_applications']) 

if st.session_state.tipo_de_analise=="Análise Algébrica":
    mdl = list(st.session_state.modelo_selecionado.values())[0]
    variables = sp.symbols(mdl['variables'])
    param_symbols = {param.strip(): sp.Symbol(param.strip()) for param in list(mdl['params'].keys())}
    equations = [sp.sympify(eq, locals=param_symbols) for eq in mdl['equations']]
    equilibria = find_symbolic_equilibria(equations, variables)
    if len(equilibria):
        equilibria = [tuple(eq.values()) for eq in equilibria]
        st.write("#### Pontos de equilíbrio do modelo")
        st.write(
            f"Os pontos de equilíbrio são os valores de ${tuple(variables)}$ onde as equações do sistema são zeradas.")

        for i, eq in enumerate(equilibria, 1):
            st.write(f"**Ponto de equilíbrio {i}:**")
            st.latex(
                f"E_{i}=\\left({sp.latex(eq[0])}, {sp.latex(eq[1])}\\right)")
        
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
            J_eval = sp.simplify(J.subs({v:k for v,k in zip(variables,eq)}))
            st.markdown(
                f"Matriz Jacobiana avaliada em $E_{i}$:")
            st.latex(f"J(E_{i}) = {sp.latex(J_eval)}")

if st.session_state.tipo_de_analise=="Análise Numérica":
    mdl = list(st.session_state.modelo_selecionado.values())[0]
    variables = sp.symbols(mdl['variables'])
    param_symbols = {param.strip(): sp.Symbol(param.strip()) for param in list(mdl['params'].keys())}
    equations = [sp.sympify(eq, locals=param_symbols) for eq in mdl['equations']]
    equilibria = find_symbolic_equilibria(equations, variables)
    equilibria = [tuple(eq.values()) for eq in equilibria]
    sb_containers[1].markdown('Condição inicial para o modelo')
    for k,v in zip(variables,mdl['initial_condition']):
        sb_containers[1].number_input(f"Condição inicial para ${sp.latex(k)}$:",0.0,value=v)

    sb_containers[1].markdown('Parâmetros do modelo')
    param_values = {}
    for k,v in mdl['params'].items():
        param_values[k] = sb_containers[1].number_input(f"Valor de ${sp.latex(sp.Symbol(k))}$:",0.0,value=v)

    numerical_equilibria = evaluate_equilibria(equilibria, param_values)
    