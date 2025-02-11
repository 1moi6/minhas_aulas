import streamlit as st
import sympy as sp

def find_symbolic_equilibria(eq1, eq2, vars):
    """
    Encontra os pontos de equilíbrio simbolicamente.
    """
    return sp.solve([eq1, eq2], vars)

def jacobian_matrix(eq1, eq2, vars):
    """
    Calcula a matriz Jacobiana simbolicamente.
    """
    J = sp.Matrix([eq1, eq2]).jacobian(vars)
    return J

def main():
    st.title("Análise de Sistemas Dinâmicos")
    
    # Definição de variáveis simbólicas
    x, y = sp.symbols('x y')
    
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
       st.write("$$\\frac{dx}{dt}$$:")
    with col2:
        eq1_input = st.text_input("Digite a primeira equação diferencial:", "r * x * (1 - x / K) + alpha * x * y", label_visibility="hidden")
    
    with col1:
       st.write("$$\\frac{dy}{dt}$$:")
    with col2:
        eq2_input = st.text_input("**$$\\frac{dx}{dt}$$:**", "s * y * (1 - y / M) + beta * x * y", label_visibility="hidden")
    
    params_input = st.text_input("Digite os parâmetros separados por vírgula:", "r, K, alpha, s, M, beta")
    
    try:
        # Criar símbolos para os parâmetros
        param_symbols = {param.strip(): sp.Symbol(param.strip()) for param in params_input.split(',') if param.strip()}
        
        # Converter as equações usando os símbolos corretamente
        eq1 = sp.sympify(eq1_input, locals=param_symbols)
        eq2 = sp.sympify(eq2_input, locals=param_symbols)
        
        # Encontrar pontos de equilíbrio simbolicamente
        equilibria = find_symbolic_equilibria(eq1, eq2, (x, y))
        
        # Calcular matriz Jacobiana simbolicamente
        J = jacobian_matrix(eq1, eq2, (x, y))
        
        # Exibir resultados no Streamlit
        st.subheader("Pontos de Equilíbrio")
        for i, eq in enumerate(equilibria, 1):
            st.write(f"Ponto de Equilíbrio {i}:")
            st.latex(f"\\left({sp.latex(eq[0])}, {sp.latex(eq[1])}\\right)")
        
        st.subheader("Matriz Jacobiana")
        st.latex(sp.latex(J))
        
    except Exception as e:
        st.error(f"Erro ao processar as equações: {e}")

if __name__ == "__main__":
    main()
