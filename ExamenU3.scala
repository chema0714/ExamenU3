//1
import org.apache.spark.sql.SparkSession
//2
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)//Quita muchos warnings
//3
val spark = SparkSession.builder().getOrCreate()
//4
import org.apache.spark.ml.clustering.KMeans
//5
//val dataset = spark.read.format("libsvm").load("Wholesale customers data.csv")
val dataset  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale customers data.csv")
//6
val feature_data = dataset.select($"Fresh",$"Milk",$"Grocery",$"Frozen",$"Detergents_Paper",$"Delicassen")
//7
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
//8
val VectorAssembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
//9
val assembler = VectorAssembler.transform(feature_data).select("features")
//10
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(assembler)
//11
val WSSE = model.computeCost(assembler)
println(s"Within set sum of Squared Errors = $WSSE")
//12
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
