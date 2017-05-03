// Databricks notebook source
// MAGIC %md
// MAGIC ## This is a notebook containing cell titles and to test the impact on cell titles when external changes are made to the notebook via git 

// COMMAND ----------

def sayHelloTo(name: String) = s"Hello $name"

// COMMAND ----------

sayHelloTo("Ganesh")

// COMMAND ----------


// support list of names
def sayHelloTo(names: List[String]) = {
   names.map(name => s"Hello $name")
}
