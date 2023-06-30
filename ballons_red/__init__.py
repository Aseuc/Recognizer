from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components




html_code = """
<!DOCTYPE html>
<html>
<head>
<style>
.balloon {
  position: absolute;
  bottom: 0;
  animation: rise 5s linear infinite;
  z-index: 999;
}

@keyframes rise {
  from {bottom: 0;}
  to {bottom: 100%;}
}
</style>
</head>
<body>

<div id="balloons"></div>

<script>
const colors = ['blue', 'pink'];
const numBalloons = 10;

for (let i = 0; i < numBalloons; i++) {
  const balloon = document.createElement('img');
  balloon.src = colors[i % colors.length] + '-balloon.png';
  balloon.className = 'balloon';
  balloon.style.left = Math.random() * 100 + '%';
  balloon.style.animationDelay = -Math.random() * 5 + 's';
  document.getElementById('balloons').appendChild(balloon);
}
</script>

</body>
</html>
"""



# Tell streamlit that there is a component called ballons_red_and_blue,
# and that the code to display that component is in the "frontend" folder
frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
	"ballons_red", path=str(frontend_dir)
)

# Create the python function that will be called
def ballons_red(
    key: Optional[str] = None,
):
    """
    Add a descriptive docstring
    """
    component_value = _component_func(
        key=key,
    )

    

    return component_value


def main():
    st.write("")
    # value = ballons_red_and_blue()

    # st.write(value)


if __name__ == "__main__":
    main()
   