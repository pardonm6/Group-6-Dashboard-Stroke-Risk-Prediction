import streamlit as st

def contact_form():
with st.form("contact_form"):
    st.text_input("First Name")
    st.text_input("Last Name")
    st.text_input("Email")
    st.text_area("Message")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.success("Thank you for reaching out! We will get back to you soon.")