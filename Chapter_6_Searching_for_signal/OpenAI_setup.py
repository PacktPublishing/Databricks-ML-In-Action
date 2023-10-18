# Databricks notebook source
import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope="dlia, key="OPENAI_API_KEY")
