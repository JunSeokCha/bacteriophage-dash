from dash import Dash

MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

external_scripts = [
    {
        'type': 'text/javascript',
        'id': 'MathJax-script',
        'src': MATHJAX_CDN,
    },
]

print('Starting App')
app = Dash(__name__,
    external_scripts=external_scripts,
    suppress_callback_exceptions = True)
app.title = 'Bacteriophage'
