package com.example.facerecognitiontest;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.example.facerecognitiontest.util.FileUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import static com.example.facerecognitiontest.App.list_label_mobile;
import static com.example.facerecognitiontest.App.list_label_pb;
import static com.example.facerecognitiontest.App.list_label_tf;
import static com.example.facerecognitiontest.App.list_vector_mobile;
import static com.example.facerecognitiontest.App.list_vector_pb;
import static com.example.facerecognitiontest.App.list_vector_tf;
import static com.example.facerecognitiontest.util.FileUtils.ROOT;

public class MainActivity extends AppCompatActivity {

    Button registration,detect,file_detection;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        registration = findViewById(R.id.registration);
        detect = findViewById(R.id.detect);
        file_detection = findViewById(R.id.file_detection);

        registration.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this,Registration.class));
            }
        });
        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this,Detection.class));
            }
        });
        file_detection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this,FilePickUp.class));
            }
        });

    }

    @Override
    protected void onStart() {
        super.onStart();
        list_vector_mobile.clear();
        list_label_mobile.clear();
        list_vector_pb.clear();
        list_label_pb.clear();
        list_vector_tf.clear();
        list_label_tf.clear();
        try {
            String name = "";
            Scanner scanner = new Scanner(new File(ROOT + File.separator + FileUtils.DATA_FILE_MOBILE));
            while (scanner.hasNextLine()) {
                String str = scanner.nextLine();
                String[] s = str.split(" ");
                float[] ed = new float[192];
                int i=0;
                for (String st : s) {
                    try {
                        String[] arr = st.split(":");
                        ed[i] = Float.parseFloat(arr[1]);
                        i++;
                    } catch (Exception e) {
                        if (!name.equals(st) || name.equals("")){
                            name = st;
                            list_label_mobile.add(name);
                        }
                    }
                }
                Log.i("test","  "+ed.length);
                list_vector_mobile.add(ed);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try {
            String name = "";
            Scanner scanner = new Scanner(new File(ROOT + File.separator + FileUtils.DATA_FILE_PB));
            while (scanner.hasNextLine()) {
                String str = scanner.nextLine();
                String[] s = str.split(" ");
                float[] ed = new float[512];
                int i=0;
                for (String st : s) {
                    try {
                        String[] arr = st.split(":");
                        ed[i] = Float.parseFloat(arr[1]);
                        i++;
                    } catch (Exception e) {
                        if (!name.equals(st) || name.equals("")){
                            name = st;
                            list_label_pb.add(name);
                        }
                    }
                }
                list_vector_pb.add(ed);
            }
            Log.i("test",list_vector_pb.size()+"  ");

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            String name = "";
            Scanner scanner = new Scanner(new File(ROOT + File.separator + FileUtils.DATA_FILE_TF));
            while (scanner.hasNextLine()) {
                String str = scanner.nextLine();
                String[] s = str.split(" ");
                float[] ed = new float[128];
                int i=0;
                for (String st : s) {
                    try {
                        String[] arr = st.split(":");
                        ed[i] = Float.parseFloat(arr[1]);
                        i++;
                    } catch (Exception e) {
                        if (!name.equals(st) || name.equals("")){
                            name = st;
                            list_label_tf.add(name);
                        }
                    }
                }
                list_vector_tf.add(ed);
            }
            Log.i("test",list_vector_tf.size()+"  ");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
