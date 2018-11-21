package com.deepika.MusicRecommendation;


import scala.collection.Seq;
import org.apache.spark.sql.expressions.WindowSpec; 
import static org.apache.spark.sql.functions.col;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.recommendation.ALS;
import static org.apache.spark.sql.functions.*;

public class ClickAnalysis {


	public static UDF1 toVector = new UDF1<Seq<Float>, Vector>(){

		public Vector call(Seq<Float> t1) throws Exception {

			List<Float> L = scala.collection.JavaConversions.seqAsJavaList(t1);
			double[] DoubleArray = new double[t1.length()]; 
			for (int i = 0 ; i < L.size(); i++) { 
				DoubleArray[i]=L.get(i); 
			} 
			return Vectors.dense(DoubleArray); 
		} 
	};

	public static void main(String[] args) {

		if(args.length < 3) {
			System.out.println("Please specify all the parameters correctly");
			return;
		}else {
			Logger.getLogger("org").setLevel(Level.ERROR);
			Logger.getLogger("akka").setLevel(Level.ERROR);

			//Create the spark session
			SparkSession spark = SparkSession.builder().config("spark.hadoop.fs.s3a.access.key", args[0])
					.config("spark.hadoop.fs.s3a.secret.key", args[1])
					.appName("MusicRecommendation").master("local[*]").getOrCreate();


			// Loads data
			System.out.println("The program is starting now..");
			System.out.println("Reading 100 MB data from S3 bucket for clickstream activity..");

			//Select only 2 columns - UserId , SongID. Ignore rows having null values
			Dataset<Row> clickStreamDataSet = spark.read().option("header", "false").csv("s3a://bigdataanalyticsupgrad/activity/sample100mb.csv").select(col("_c0").as("UserID"),col("_c2").as("SongID")).na().drop();
			//clickStreamDataSet.show();

			//Calculating Frequency from this data based on UserID and SongID
			System.out.println("Calculating frequency now..");
			Dataset<Row> datasetFreq = clickStreamDataSet.groupBy("UserID", "SongID")
					.agg(functions.count("*").alias("Frequency"));

			//This will convert the String values of UserID to numeric
			StringIndexer indexer1 = new StringIndexer()
					.setInputCol("UserID").setOutputCol("userIndex");

			System.out.println("Changing the String column UserID to numeric index now..");
			Dataset<Row> indexed_first = indexer1.fit(datasetFreq).transform(datasetFreq);

			//This will convert the String values of SongID to numeric
			StringIndexer indexer2 = new StringIndexer()
					.setInputCol("SongID").setOutputCol("songIndex");

			System.out.println("Changing the String column SongID to numeric index now..");
			Dataset<Row> indexed_final = indexer2.fit(indexed_first).transform(indexed_first);


			System.out.println("Starting the ALS algorithm now for getting the implicit feedback...");
			ALS als = new ALS()
					.setRank(10)
					.setMaxIter(5)
					.setImplicitPrefs(true)
					.setUserCol("userIndex")
					.setItemCol("songIndex")
					.setRatingCol("Frequency");

			System.out.println("Fitting the dataset into ALS model now...");
			ALSModel model = als.fit(indexed_final);
			Dataset<Row> userFactors = model.userFactors();

			//Recieved the implicit factors from ALS. However, it is in array format
			//This needs to be changed into vector

			System.out.println("Recieved the implicit factors from ALS. However, it is in array format. Working to change it into Vector now...");
			spark.udf().register("toVector", toVector, new VectorUDT());
			Dataset<Row> ds2 = userFactors.withColumn("features", functions.callUDF("toVector",userFactors.col("features")));


			//Setting K as 300 for K-means algorithm
			KMeans kmeansFinal = new KMeans().setK(300).setSeed(1L);

			System.out.println("Fitting the dataset into K-means algorithm now");
			KMeansModel modelFinal = kmeansFinal.fit(ds2);

			Dataset<Row> predictionsTest = modelFinal.transform(ds2);

			//Join predictions and indexed_1 as we need to convert back numeric userIndex to String UserID
			Dataset<Row> joinedDS1 = predictionsTest
					.join(indexed_first, predictionsTest.col("id").equalTo(indexed_first.col("userIndex")), "inner")
					.drop(indexed_first.col("userIndex"));
			System.out.println("Showing the prediction clusters for each user based on feature vector...");
			joinedDS1.show();


			//Loads metadata
			System.out.println("Reading the metadata now...");
			Dataset<Row> metadataNew = spark.read().option("header", "false").csv("s3a://bigdataanalyticsupgrad/newmetadata/*").select(col("_c0").as("SongID"),col("_c1").as("ArtistID")).na().drop();
			//metadataNew.show();


			System.out.println("Working on the final predicted outcome");
			// Joins the two datasets - joinedDS1 and metadataNew
			//DS2 contains 4 columns finally - SongID, ArtistID , prediction(cluster) and UserID
			Dataset<Row> joinedDS2 = metadataNew
					.join(joinedDS1, metadataNew.col("SongID").equalTo(joinedDS1.col("SongID")), "inner")
					.drop(joinedDS1.col("SongID"))
					.select("SongID" , "ArtistID" , "prediction" , "UserID");

			System.out.println("Recieved the final predicted outcome after Training the model.");

			//Group by on prediction and UserID to get groups based on these 2 columns
			Dataset<Row> countOfUsers1 = joinedDS2.groupBy("prediction" , "UserID").count().as("cluster-userCount");
			System.out.println("Calculating the count of unique users in each cluster..");

			//Further group by on Prediction to get distinct users in each prediction / cluster
			Dataset<Row> countOfUsersFinal = countOfUsers1.groupBy("prediction")
					.agg(functions.count("*").alias("CountOfUsers"));
			countOfUsersFinal.show();


			//Group by on Prediction and ArtistID to get count of each cluster and artist id pair
			Dataset<Row> popularArtist1 = joinedDS2.groupBy("prediction", "ArtistID")
					.agg(functions.count("*").alias("ArtistPopularity"));


			//Select the most popular artist from each cluster
			System.out.println("Coming up with most popular artist in each cluster...");
			WindowSpec w = org.apache.spark.sql.expressions.Window.partitionBy("prediction").orderBy(functions.desc("ArtistPopularity"));
			Dataset<Row> popularArtistFinal = popularArtist1.withColumn("rn", row_number().over(w))
					.where("rn = 1")
					.select(popularArtist1.col("prediction") , popularArtist1.col("ArtistID"));
			popularArtistFinal.show();

			System.out.println("Reading Notification clicks data now for validation....");
			Dataset<Row> notificationClickData = spark.read().option("header", "false").csv("s3a://bigdataanalyticsupgrad/notification_clicks/*")
					.select(col("_c0").as("NotificationID"),
							col("_c1").as("UserID")).na().drop();
			//notificationClickData.show();

			System.out.println("Reading Notification Artist data now for validation....");
			Dataset<Row> notificationArtistData = spark.read().option("header", "false").csv("s3a://bigdataanalyticsupgrad/notification_actor/*")
					.select(col("_c0").as("NotificationID"),
							col("_c1").as("ArtistID")).na().drop();
			//notificationArtistData.show();


			System.out.println("Finding the ArtistID , NotificationID for each predicted cluster..");
			Dataset<Row> joinedDS4 = notificationArtistData
					.join(popularArtistFinal, notificationArtistData.col("ArtistID").equalTo(popularArtistFinal.col("ArtistID")), "inner")
					.drop(popularArtistFinal.col("ArtistID"));
			joinedDS4.show();


			//DS2 contains 4 columns finally - SongID, ArtistID , prediction(cluster) and UserID
			//DS4 will contain 3 columns - ArtistId , prediction and NotificationID
			//remove columns - songid , prediction , ArtistID . Fields to keep - UserId and NotificationID
			//We need to group this by NotificationID so that we know how many times it was clicked . 
			//Later group by NotificationID 

			System.out.println("Calculations for UserIDs in predicted clusters to which Notification would be sent is in progress..");
			Dataset<Row> joinedDS5 = joinedDS4
					.join(joinedDS2, joinedDS4.col("ArtistID").equalTo(joinedDS2.col("ArtistID"))
							.and(joinedDS4.col("prediction").equalTo(joinedDS2.col("prediction"))), "inner")
					.drop(joinedDS4.col("prediction")).drop(joinedDS4.col("ArtistID"))
					.select("UserID" , "NotificationID");

			//Group by on notification to find the targeted users who will receive the notification
			System.out.println("Predicted Count of users recieving the notification...");
			Dataset<Row> DS5 = joinedDS5.groupBy("NotificationID")
					.agg(functions.count("*").alias("PredictedCount"));

			DS5.show();

			//We have the notification click information from our validation data set
			//joinedDS5 has UserID and NotificationID
			//We will join these 2 now to get the actual users who clicked the notification
			//pick only 2 cols - userid and notificationid 

			System.out.println("Calculations for UserIDs who actually clicked on Notification is in progress..");
			Dataset<Row> joinedDS7 = joinedDS5
					.join(notificationClickData, joinedDS5.col("UserID").equalTo(notificationClickData.col("UserID"))
							.and(joinedDS5.col("NotificationID").equalTo(notificationClickData.col("NotificationID"))), "inner")
					.drop(joinedDS5.col("UserID")).drop(joinedDS5.col("NotificationID"));


			//Group by on notification to find the actual users who have clicked the notification
			System.out.println("Actual Count of users who clicked on the notification ...");
			Dataset<Row> DS7 = joinedDS7.groupBy("NotificationID")
					.agg(functions.count("*").alias("ActualCount"));


			//Joining the two datasets DS7 and DS5 based on notificationID
			System.out.println("System is in progress to calculate CTR for each notification");
			Dataset<Row> DS8 = DS5.join(DS7 , DS5.col("NotificationID").equalTo(DS7.col("NotificationID")) , "inner")
					.drop(DS5.col("NotificationID"));

			System.out.println("Sit tight. Clickthrough rate for each notification based on Actual/Targeted clicks is as below: ");
			Dataset<Row> CTR = DS8.withColumn("CTR", col("ActualCount").divide(col("PredictedCount")));

			CTR.show();

			System.out.println("Saving the result in CSV now...");
			Dataset<Row> CTRPrint = CTR.select("NotificationID" , "CTR");
			CTRPrint.coalesce(1).write().mode(SaveMode.Overwrite).format("csv").save(args[2] + "/" + "CTR");

			System.out.println("The program is ending now. Exiting...");
			spark.stop();
		}
	}


}
