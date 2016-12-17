// Databricks notebook source
// MAGIC %md # Example 1

// COMMAND ----------

// MAGIC %md ### Create a Table Strucuture

// COMMAND ----------

import org.apache.spark.sql.functions._
val employee = spark.range(0, 20).select($"id".as("employee_id"), (rand() * 3).cast("int").as("dep_id"), (rand() * 40 + 20).cast("int").as("age"))
// employee.printSchema()

// COMMAND ----------

employee.write.mode("overwrite").partitionBy("employee_id").save("/mnt/vgiri/newparts")

// COMMAND ----------

// MAGIC %fs ls /mnt/vgiri/newparts

// COMMAND ----------

// spark.sql(s"""
// create table part3 (dep_id int,age int) partitioned by(employee_id string) 
// stored as parquet
// location '/mnt/vgiri/newparts'
// """)

spark.sql(s"""
create table part (employee_id int, dep_id int,age int) 
using PARQUET
options ('path' ="/mnt/vgiri/newparts")
partitioned by (employee_id)
""")

// COMMAND ----------

// MAGIC %sql
// MAGIC create table part2 (dep_id int,age int) partitioned by(employee_id string) 
// MAGIC stored as parquet
// MAGIC location '/mnt/vgiri/newparts'

// COMMAND ----------

// MAGIC %sql
// MAGIC create table part4 (employee_id int, dep_id int,age int) 
// MAGIC using PARQUET
// MAGIC options ('path' ="/mnt/vgiri/newparts")
// MAGIC partitioned by (employee_id)

// COMMAND ----------

employee.repartition(10000000000).rdd.partitions.size

// COMMAND ----------

// MAGIC %md ### Repartition and Save to S3

// COMMAND ----------

employee.write.mode("overwrite").format("json").save("/mnt/vgiri/testsmallfiles")

// COMMAND ----------

// MAGIC %md ##### Check the Original # of Partitions or Splits

// COMMAND ----------

dbutils.fs.ls("/mnt/vgiri/testsmallfiles").size

// COMMAND ----------

import org.apache.spark.sql.types._

val schema = StructType(Array(StructField("employee_id",StringType,true),StructField("dept_id",StringType,true),StructField("age",StringType,true)))

// COMMAND ----------

val readSmallFiles = spark.read.schema(schema).json("/mnt/vgiri/testsmallfiles")

// COMMAND ----------

// MAGIC %md ### Reduced After Spark read the Small Files 70% rate

// COMMAND ----------

readSmallFiles.rdd.partitions.size

// COMMAND ----------

// MAGIC %md # Example 2

// COMMAND ----------

case class Employee(id: Int, name: String, designation: String, age: Int, departmentId: Int, manager: String =null, managerId:Int=0)

// COMMAND ----------

val employees = Seq(
Employee(1001,"BLAKE", "CEO", 45,1),
Employee(1002,"CLARK", "Director of Engineering", 32,2,"BLAKE",1001) ,
Employee(1003,"JONES", "VP Sales", 45, 3, "BLAKE" ,1001) ,
Employee(1004,"MARTIN","Engineering Manager", 28,2,"CLARK" ,1002), 
Employee(1005,"ALLEN", "Software Engineer", 23,2,"MARTIN",1004) ,
Employee(1006,"TURNER","Software Engineer", 23,2,"MARTIN" ,1004), 
Employee(1007,"JAMES", "Sales Engineer",21,3,"JONES" ,1003)
)

// COMMAND ----------

val empDF = employees.toDF

// COMMAND ----------

empDF.printSchema

// COMMAND ----------

// MAGIC %md #### Convert DF to DS

// COMMAND ----------

val empDs = empDF.as[Employee]

// COMMAND ----------

empDs.map(_.id)

// COMMAND ----------

// MAGIC %md ##### won't compile

// COMMAND ----------

empDs.map(_.id1)

// COMMAND ----------

res12.show

// COMMAND ----------

// MAGIC %md #### Perform map transformation on DS returns DataSet

// COMMAND ----------

val empdsreturn = empDs.map( x => List(x.id,"Transformation Inside a DF",x.designation))

// COMMAND ----------

val empdsreturn = empDs.map( x => List(x.id.toString,"Transformation Inside a DF",x.designation))

// COMMAND ----------

empdsreturn.count

// COMMAND ----------

empdsreturn.toDF

// COMMAND ----------

// MAGIC %md # Example 3 (Showing Predicate SubQueries are supported)

// COMMAND ----------

import org.apache.spark.sql.functions._
val employee = sqlContext.range(0, 10).select($"id".as("employee_id"), (rand() * 3).cast("int").as("dep_id"), (rand() * 40 + 20).cast("int").as("age"))
val visit = sqlContext.range(0, 100).select($"id".as("visit_id"), when(rand() < 0.95, ($"id" % 8)).as("employee_id"))
val appointment = sqlContext.range(0, 100).select($"id".as("appointment_id"), when(rand() < 0.95, ($"id" % 7)).as("employee_id"))
employee.createOrReplaceTempView("employee")
visit.createOrReplaceTempView("visit")
appointment.createOrReplaceTempView("appointment")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT  *
// MAGIC FROM    employee
// MAGIC WHERE   employee_id IN (SELECT  employee_id
// MAGIC                         FROM    visit)

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT  *
// MAGIC FROM    employee
// MAGIC WHERE   employee_id NOT IN (SELECT  employee_id
// MAGIC                             FROM    visit)

// COMMAND ----------

// MAGIC %md # Example 4

// COMMAND ----------

// MAGIC %sql drop table if exists vectortesting_2_0

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "hour", "mobile", "userFeatures", "clicked")

val assembler = new VectorAssembler()
  .setInputCols(Array("hour", "mobile", "userFeatures"))
  .setOutputCol("features")

val output = assembler.transform(dataset)
output.write.saveAsTable("vectortesting_2_0")

// COMMAND ----------

// MAGIC %md ## Say you are trying to access the mlib.linalg.Vectors column in 2.0
// MAGIC (Fails in 2.0)

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.sql("select id,hour,mobile,userFeatures,clicked from vectortesting_1_6_2")
val assembler = new VectorAssembler().setInputCols(Array("hour", "mobile", "userFeatures")).setOutputCol("features")
val output = assembler.transform(dataset)
display(output)

// COMMAND ----------

// MAGIC %md ## So, use this way to access the mlib.linalg to convert to ml.lialg

// COMMAND ----------

import org.apache.spark.mllib.linalg.Vector
val convertToML = udf { v: Vector => v.asML }

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.sql("select id,hour,mobile,userFeatures,clicked from vectortesting_1_6_2").withColumn("userFeatures", convertToML('userFeatures))
val assembler = new VectorAssembler().setInputCols(Array("hour", "mobile", "userFeatures")).setOutputCol("features")
val output = assembler.transform(dataset)
display(output)

// COMMAND ----------

// MAGIC %sql
// MAGIC set hive.cli.print.current.db=true

// COMMAND ----------

