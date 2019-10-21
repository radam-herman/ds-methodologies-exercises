Estimating Home Values for Zillow data science team project.

Objective -
predict the values of single unit properties that the tax district assesses using the property data from those whose last transaction was during the "hot months" (in terms of real estate demand) of May and June in 2017.

Side information - tax rate distribution by county (and state)


Products contained in this project-
  1) The detailed process is contained and outlined in the project.ipynb file.
  2) A report in the form of a slide presentation, called "Zillow DS Team Briefing"
  3) The project.ipynb also requires a Python document containing functions for its use.
      That Python file is project_wrangle.py
  4) last but not least, the Zillow Data Set being used is reachable within the
     project.ipynb via importing the project_wrangle.py and calling the function on its
     own line, like this:
         df = wrangle_project.wrangle_zillow_bl()

       ** important **
     The project_wrangle.py file is dependent on an env.py file which contains three
     important items in order to connect to and retrieve the data set:

          host = "111.111.111.111"
          user = "my_username_to_server"
          password = "myP@ssw0rdToTheServer"

     You will need to make an env.py file with the above three lines, providing your specific information between the quotes - host (aka IP of the server hosting the data), user (your username), and password (your password). And note, the quotes are necessary and kept so only change the information within the quotes. 
